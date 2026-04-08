pub mod ghcp;

use crate::openai::types::{
    ChatCompletionMessageToolCall, CreateChatCompletionRequest, CreateEmbeddingRequest,
    EmbeddingObject,
};
use std::future::Future;

#[derive(Debug)]
pub struct ProviderChatResponse {
    pub model: String,
    pub content: Option<String>,
    pub tool_calls: Vec<ChatCompletionMessageToolCall>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug)]
pub struct ProviderEmbeddingResponse {
    pub model: String,
    pub data: Vec<EmbeddingObject>,
    pub prompt_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("operation not supported: {0}")]
    NotSupported(String),
    #[error("upstream provider error: {0}")]
    Upstream(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

pub trait ModelProvider {
    fn create_chat_completion(
        &self,
        request: CreateChatCompletionRequest,
        default_model: Option<&str>,
    ) -> impl Future<Output = Result<ProviderChatResponse, ProviderError>> + Send;

    fn create_embeddings(
        &self,
        _request: CreateEmbeddingRequest,
        _default_model: Option<&str>,
    ) -> impl Future<Output = Result<ProviderEmbeddingResponse, ProviderError>> + Send {
        std::future::ready(Err(ProviderError::NotSupported(
            "embeddings are not available for GHCP yet".to_string(),
        )))
    }
}
