use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Anthropic Messages API request.
///
/// Many fields are accepted but pass through to / are mapped onto the OpenAI
/// chat completion that we ultimately dispatch. We deserialize via `Value` for
/// `messages`, `system`, `tools`, and `tool_choice` because Anthropic's content
/// block schema is open-ended (text, image, tool_use, tool_result, thinking,
/// document, ...) and we want to forward unknown shapes rather than reject.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CreateMessageRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u64>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub thinking: Option<Value>,
}

/// Anthropic Messages API response.
#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Serialize, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}
