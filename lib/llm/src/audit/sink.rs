// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;
use async_nats::jetstream;
use async_trait::async_trait;
use dynamo_runtime::transports::nats;
use std::sync::Arc;
use tokio::fs::{self, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::broadcast;
use tokio::sync::Mutex;

use super::{bus, handle::AuditRecord};

#[async_trait]
pub trait AuditSink: Send + Sync {
    fn name(&self) -> &'static str;
    async fn emit(&self, rec: &AuditRecord);
}

pub struct StderrSink;
#[async_trait]
impl AuditSink for StderrSink {
    fn name(&self) -> &'static str {
        "stderr"
    }
    async fn emit(&self, rec: &AuditRecord) {
        match serde_json::to_string(rec) {
            Ok(js) => {
                tracing::info!(target="dynamo_llm::audit", log_type="audit", record=%js, "audit")
            }
            Err(e) => tracing::warn!("audit: serialize failed: {e}"),
        }
    }
}

pub struct NatsSink {
    js: jetstream::Context,
    subject: String,
}

impl NatsSink {
    pub fn new(nats_client: dynamo_runtime::transports::nats::Client) -> Self {
        let subject = std::env::var("DYN_AUDIT_NATS_SUBJECT")
            .unwrap_or_else(|_| "dynamo.audit.v1".to_string());
        Self {
            js: nats_client.jetstream().clone(),
            subject,
        }
    }
}

pub struct FileSink {
    writer: Mutex<BufWriter<tokio::fs::File>>,
}

impl FileSink {
    pub async fn new_from_env() -> anyhow::Result<Self> {
        let path = std::env::var("DYN_AUDIT_FILE_PATH")
            .context("DYN_AUDIT_FILE_PATH is required when DYN_AUDIT_SINKS includes 'file'")?;
        Self::new(path).await
    }

    async fn new(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).await.with_context(|| {
                format!("Failed to create audit log directory {}", parent.display())
            })?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .with_context(|| format!("Failed to open audit log file {}", path.display()))?;

        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }
}

#[async_trait]
impl AuditSink for FileSink {
    fn name(&self) -> &'static str {
        "file"
    }

    async fn emit(&self, rec: &AuditRecord) {
        let json = match serde_json::to_vec(rec) {
            Ok(json) => json,
            Err(e) => {
                tracing::warn!("file audit sink: serialize failed: {e}");
                return;
            }
        };

        let mut writer = self.writer.lock().await;
        if let Err(e) = writer.write_all(&json).await {
            tracing::warn!("file audit sink: write failed: {e}");
            return;
        }
        if let Err(e) = writer.write_all(b"\n").await {
            tracing::warn!("file audit sink: newline write failed: {e}");
            return;
        }
        if let Err(e) = writer.flush().await {
            tracing::warn!("file audit sink: flush failed: {e}");
        }
    }
}

#[async_trait]
impl AuditSink for NatsSink {
    fn name(&self) -> &'static str {
        "nats"
    }

    async fn emit(&self, rec: &AuditRecord) {
        match serde_json::to_vec(rec) {
            Ok(bytes) => {
                if let Err(e) = self.js.publish(self.subject.clone(), bytes.into()).await {
                    tracing::warn!("nats: publish failed: {e}");
                }
            }
            Err(e) => tracing::warn!("nats: serialize failed: {e}"),
        }
    }
}

async fn parse_sinks_from_env() -> anyhow::Result<Vec<Arc<dyn AuditSink>>> {
    let cfg = std::env::var("DYN_AUDIT_SINKS").unwrap_or_else(|_| "stderr".into());
    let mut out: Vec<Arc<dyn AuditSink>> = Vec::new();
    for name in cfg.split(',').map(|s| s.trim().to_lowercase()) {
        match name.as_str() {
            "stderr" | "" => out.push(Arc::new(StderrSink)),
            "file" => out.push(Arc::new(FileSink::new_from_env().await?)),
            "nats" => {
                let nats_client = nats::ClientOptions::default()
                    .connect()
                    .await
                    .context("Attempting to connect NATS sink from env var DYN_AUDIT_SINKS")?;
                out.push(Arc::new(NatsSink::new(nats_client)));
            }
            // "pg"   => out.push(Arc::new(PostgresSink::from_env())),
            other => tracing::warn!(%other, "audit: unknown sink ignored"),
        }
    }
    Ok(out)
}

/// spawn one worker per sink; each subscribes to the bus (off hot path)
pub async fn spawn_workers_from_env() -> anyhow::Result<()> {
    let sinks = parse_sinks_from_env().await?;
    for sink in sinks {
        let name = sink.name();
        let mut rx: broadcast::Receiver<AuditRecord> = bus::subscribe();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(rec) => sink.emit(&rec).await,
                    Err(broadcast::error::RecvError::Lagged(n)) => tracing::warn!(
                        sink = name,
                        dropped = n,
                        "audit bus lagged; dropped records"
                    ),
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }
    tracing::info!("Audit sinks ready.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use tempfile::tempdir;

    fn create_test_request(model: &str) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test message"}],
            "store": true
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    #[tokio::test]
    async fn test_file_sink_writes_jsonl_record() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = dir.path().join("audit").join("requests.jsonl");
        let sink = FileSink::new(&path).await.expect("failed to create file sink");
        let rec = AuditRecord {
            schema_version: 1,
            request_id: "req-1".to_string(),
            requested_streaming: false,
            model: "Kimi-K2.5".to_string(),
            request: Some(Arc::new(create_test_request("Kimi-K2.5"))),
            response: None,
        };

        sink.emit(&rec).await;

        let content = tokio::fs::read_to_string(&path)
            .await
            .expect("failed to read audit file");
        let mut lines = content.lines();
        let line = lines.next().expect("missing audit line");
        let value: serde_json::Value =
            serde_json::from_str(line).expect("failed to parse audit line as json");

        assert_eq!(value["request_id"], "req-1");
        assert_eq!(value["model"], "Kimi-K2.5");
        assert_eq!(value["request"]["model"], "Kimi-K2.5");
        assert!(lines.next().is_none(), "expected exactly one audit line");
    }
}
