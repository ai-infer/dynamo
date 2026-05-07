// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde_json::Value;

/// Build model-aware chat template args for thinking mode.
///
/// Policy:
/// - Known parser/model families: map only the keys those templates commonly use.
/// - Unknown/unverified families: leave template args unchanged. Some models do
///   not support switching thinking mode, and adding compatibility keys can
///   change their prompt rendering unexpectedly.
pub(super) fn build_model_aware_thinking_args(
    model_name: &str,
    reasoning_parser: Option<&str>,
    enable_thinking: bool,
) -> HashMap<String, Value> {
    let thinking = Value::Bool(enable_thinking);
    let name = model_name.to_ascii_lowercase();

    match reasoning_parser {
        Some("kimi_k25") => {
            let mut args = HashMap::new();
            args.insert("thinking".to_string(), thinking);
            return args;
        }
        Some("deepseek_r1") => {
            let mut args = HashMap::new();
            args.insert("thinking".to_string(), thinking);
            return args;
        }
        Some("nemotron_nano") | Some("nemotron3") => {
            let mut args = HashMap::new();
            args.insert("enable_thinking".to_string(), thinking);
            return args;
        }
        _ => {}
    }

    // Known model families where parser may be absent from runtime config.
    if name.contains("kimi") {
        let mut args = HashMap::new();
        args.insert("thinking".to_string(), thinking);
        return args;
    }
    if name.contains("deepseek") && name.contains("v3.2") {
        let mut args = HashMap::new();
        args.insert("thinking".to_string(), thinking);
        return args;
    }
    if name.contains("deepseek") && name.contains("r1") {
        let mut args = HashMap::new();
        args.insert("thinking".to_string(), thinking);
        return args;
    }
    if name.contains("glm-5.1") || name.contains("glm5.1") {
        let mut args = HashMap::new();
        args.insert("enable_thinking".to_string(), thinking);
        return args;
    }

    HashMap::new()
}

pub(super) fn merge_thinking_args(
    chat_template_args: &mut Option<HashMap<String, Value>>,
    model_name: &str,
    reasoning_parser: Option<&str>,
    enable_thinking: bool,
) -> bool {
    let mapped = build_model_aware_thinking_args(model_name, reasoning_parser, enable_thinking);
    if mapped.is_empty() {
        return false;
    }
    let args = chat_template_args.get_or_insert_with(HashMap::new);
    for (k, v) in mapped {
        args.insert(k, v);
    }
    true
}

pub(super) fn merge_default_thinking_args_if_absent(
    chat_template_args: &mut Option<HashMap<String, Value>>,
    model_name: &str,
    reasoning_parser: Option<&str>,
) -> bool {
    let mapped = build_model_aware_thinking_args(model_name, reasoning_parser, true);
    if mapped.is_empty() {
        return false;
    }
    let args = chat_template_args.get_or_insert_with(HashMap::new);
    for (k, v) in mapped {
        args.entry(k).or_insert(v);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kimi_k25_parser_uses_thinking_key_only() {
        let args =
            build_model_aware_thinking_args("moonshotai/Kimi-K2.5", Some("kimi_k25"), true);
        assert_eq!(args.get("thinking"), Some(&Value::Bool(true)));
        assert!(!args.contains_key("thinking_mode"));
        assert!(!args.contains_key("enable_thinking"));
    }

    #[test]
    fn test_deepseek_r1_parser_uses_thinking_key_only() {
        let args = build_model_aware_thinking_args(
            "deepseek-ai/DeepSeek-R1-0528",
            Some("deepseek_r1"),
            false,
        );
        assert_eq!(args.get("thinking"), Some(&Value::Bool(false)));
        assert!(!args.contains_key("thinking_mode"));
        assert!(!args.contains_key("enable_thinking"));
    }

    #[test]
    fn test_deepseek_v32_model_name_uses_thinking_key_only() {
        let args = build_model_aware_thinking_args("nvidia/DeepSeek-V3.2-NVFP4", None, true);
        assert_eq!(args.get("thinking"), Some(&Value::Bool(true)));
        assert!(!args.contains_key("thinking_mode"));
        assert!(!args.contains_key("enable_thinking"));
    }

    #[test]
    fn test_glm51_model_name_uses_enable_thinking_only() {
        let args = build_model_aware_thinking_args("zai-org/GLM-5.1", Some("glm45"), true);
        assert_eq!(args.get("enable_thinking"), Some(&Value::Bool(true)));
        assert!(!args.contains_key("thinking"));
        assert!(!args.contains_key("thinking_mode"));
    }

    #[test]
    fn test_unknown_model_leaves_thinking_args_unchanged() {
        let args = build_model_aware_thinking_args("unknown/model", Some("basic"), true);
        assert!(args.is_empty());
    }

    #[test]
    fn test_unknown_model_merge_is_noop() {
        let mut chat_template_args = None;
        let merged = merge_thinking_args(
            &mut chat_template_args,
            "unknown/model",
            Some("basic"),
            true,
        );

        assert!(!merged);
        assert!(chat_template_args.is_none());
    }
}
