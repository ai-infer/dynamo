// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_config::env_is_truthy;

#[derive(Clone, Copy)]
pub struct AuditPolicy {
    pub enabled: bool,
    pub force_logging: bool,
}

static POLICY: OnceLock<AuditPolicy> = OnceLock::new();

/// Audit is enabled if we have at least one sink
pub fn init_from_env() -> AuditPolicy {
    AuditPolicy {
        enabled: std::env::var("DYN_AUDIT_SINKS").is_ok(),
        force_logging: env_is_truthy("DYN_AUDIT_FORCE_LOGGING"),
    }
}

pub fn policy() -> AuditPolicy {
    *POLICY.get_or_init(init_from_env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use temp_env::with_vars;

    #[test]
    fn test_force_logging_accepts_truthy_env_values() {
        with_vars(
            [
                ("DYN_AUDIT_SINKS", Some("file")),
                ("DYN_AUDIT_FORCE_LOGGING", Some("1")),
            ],
            || {
                let policy = init_from_env();
                assert!(policy.enabled);
                assert!(policy.force_logging);
            },
        );
    }
}
