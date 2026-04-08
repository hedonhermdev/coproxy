use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatCompletionRequestMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[serde(default)]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequestMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionTool {
    #[serde(rename = "type", default = "default_function_tool_type")]
    pub kind: String,
    pub function: FunctionObject,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionObject {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatCompletionToolChoiceOption {
    String(String),
    Named { function: FunctionNameOnly },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionNameOnly {
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatCompletionMessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct CreateChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: CompletionUsage,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponseMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatCompletionMessageToolCall>,
}

#[derive(Debug, Default, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Serialize)]
pub struct ListModelsResponse {
    pub object: &'static str,
    pub data: Vec<Model>,
}

#[derive(Debug, Serialize)]
pub struct Model {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateEmbeddingRequest {
    pub model: Option<String>,
    pub input: Value,
}

fn default_function_tool_type() -> String {
    "function".to_string()
}

#[derive(Debug, Serialize)]
pub struct CreateEmbeddingResponse {
    pub object: &'static str,
    pub model: String,
    pub data: Vec<EmbeddingObject>,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Default, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
}
