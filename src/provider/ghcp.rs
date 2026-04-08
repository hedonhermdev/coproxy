use crate::auth::device_flow;
use crate::auth::token_store::{GhcpTokenRecord, TokenStore};
use crate::openai::types::{ChatCompletionMessageToolCall, CreateChatCompletionRequest};
use crate::provider::{ModelProvider, ProviderChatResponse, ProviderError};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

const GITHUB_API_KEY_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const DEFAULT_GHCP_API: &str = "https://api.githubcopilot.com";
const DEFAULT_MODEL: &str = "gpt-4o";

#[derive(Clone)]
pub struct GhcpProvider {
    client: Client,
    store: TokenStore,
    github_token_override: Option<String>,
    cached_ghcp_token: Arc<Mutex<Option<CachedToken>>>,
    cached_model_list: Arc<Mutex<Option<CachedModels>>>,
}

#[derive(Clone, Debug)]
struct CachedToken {
    token: String,
    expires_at: i64,
    api_endpoint: String,
}

#[derive(Clone, Debug)]
struct CachedModels {
    models: Vec<String>,
    fetched_at: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct GhcpApiKeyResponse {
    token: String,
    expires_at: i64,
    #[serde(default)]
    endpoints: Option<ApiEndpoints>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ApiEndpoints {
    #[serde(default)]
    api: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpstreamChatResponse {
    model: Option<String>,
    #[serde(default)]
    choices: Vec<UpstreamChoice>,
    #[serde(default)]
    usage: Option<UpstreamUsage>,
}

#[derive(Debug, Deserialize)]
struct UpstreamChoice {
    message: UpstreamMessage,
}

#[derive(Debug, Deserialize)]
struct UpstreamMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
}

#[derive(Debug, Deserialize)]
struct UpstreamUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct UpstreamModelsResponse {
    #[serde(default)]
    data: Vec<UpstreamModel>,
}

#[derive(Debug, Deserialize)]
struct UpstreamModel {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model_picker_enabled: Option<bool>,
}

impl GhcpProvider {
    pub fn new(store: TokenStore, github_token_override: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            store,
            github_token_override,
            cached_ghcp_token: Arc::new(Mutex::new(None)),
            cached_model_list: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn ensure_ready(&self, allow_device_login: bool) -> anyhow::Result<()> {
        let _ = self.resolve_ghcp_token(allow_device_login).await?;
        Ok(())
    }

    pub fn model_catalog(&self, default_model: Option<&str>) -> Vec<String> {
        let mut models = vec![
            DEFAULT_MODEL.to_string(),
            "gpt-4.1".to_string(),
            "gpt-4.1-mini".to_string(),
            "o3-mini".to_string(),
            "claude-3.5-sonnet".to_string(),
        ];
        if let Some(custom) = default_model {
            let trimmed = custom.trim();
            if !trimmed.is_empty() && !models.iter().any(|m| m == trimmed) {
                models.insert(0, trimmed.to_string());
            }
        }
        models
    }

    pub async fn list_available_models(
        &self,
        default_model: Option<&str>,
    ) -> Result<Vec<String>, ProviderError> {
        if let Some(cached) = self.cached_models_if_fresh().await {
            return Ok(merge_with_default(cached, default_model));
        }

        match self.fetch_models_from_upstream().await {
            Ok(models) => {
                let now = chrono::Utc::now().timestamp();
                let mut lock = self.cached_model_list.lock().await;
                *lock = Some(CachedModels {
                    models: models.clone(),
                    fetched_at: now,
                });
                Ok(merge_with_default(models, default_model))
            }
            Err(error) => {
                tracing::warn!("Falling back to static model catalog: {}", error);
                Ok(self.model_catalog(default_model))
            }
        }
    }

    async fn cached_models_if_fresh(&self) -> Option<Vec<String>> {
        let lock = self.cached_model_list.lock().await;
        let cached = lock.as_ref()?;
        let now = chrono::Utc::now().timestamp();
        if now - cached.fetched_at > 300 {
            return None;
        }
        Some(cached.models.clone())
    }

    async fn fetch_models_from_upstream(&self) -> Result<Vec<String>, ProviderError> {
        let creds = self
            .resolve_ghcp_token(false)
            .await
            .map_err(|error| ProviderError::Unauthorized(error.to_string()))?;

        let url = format!("{}/models", creds.api_endpoint.trim_end_matches('/'));
        let mut req = self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", creds.token));

        for (header, value) in copilot_headers() {
            req = req.header(header, value);
        }

        let response = req.send().await.map_err(|error| {
            ProviderError::Upstream(format!("failed calling GHCP models endpoint: {error}"))
        })?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED
            || response.status() == reqwest::StatusCode::FORBIDDEN
        {
            self.invalidate_ghcp_token_cache().await;
            return Err(ProviderError::Unauthorized(
                "GHCP token expired or invalid".to_string(),
            ));
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Upstream(format!(
                "GHCP models request failed ({status}): {}",
                sanitize_error_body(&body)
            )));
        }

        let parsed = response
            .json::<UpstreamModelsResponse>()
            .await
            .map_err(|error| {
                ProviderError::Upstream(format!("failed parsing GHCP models response: {error}"))
            })?;

        let mut models = parsed
            .data
            .into_iter()
            .filter(|entry| !matches!(entry.model_picker_enabled, Some(false)))
            .filter_map(|entry| entry.id)
            .collect::<Vec<_>>();

        models.extend(self.model_catalog(None));
        models.sort();
        models.dedup();
        Ok(models)
    }

    async fn resolve_ghcp_token(&self, allow_device_login: bool) -> anyhow::Result<CachedToken> {
        let mut lock = self.cached_ghcp_token.lock().await;

        if let Some(cached) = lock.as_ref() {
            if is_token_fresh(cached.expires_at) {
                return Ok(cached.clone());
            }
        }

        if let Some(stored) = self.store.load_ghcp_token().await? {
            if is_token_fresh(stored.expires_at) {
                let cached = CachedToken {
                    token: stored.token,
                    expires_at: stored.expires_at,
                    api_endpoint: stored.api_endpoint,
                };
                *lock = Some(cached.clone());
                return Ok(cached);
            }
        }

        let github_access_token = self.resolve_github_access_token(allow_device_login).await?;
        let exchanged = self.exchange_github_for_ghcp(&github_access_token).await?;
        let endpoint = exchanged
            .endpoints
            .and_then(|e| e.api)
            .unwrap_or_else(|| DEFAULT_GHCP_API.to_string());

        let record = GhcpTokenRecord {
            token: exchanged.token.clone(),
            expires_at: exchanged.expires_at,
            api_endpoint: endpoint.clone(),
        };
        self.store.save_ghcp_token(&record).await?;

        let cached = CachedToken {
            token: exchanged.token,
            expires_at: exchanged.expires_at,
            api_endpoint: endpoint,
        };
        *lock = Some(cached.clone());
        Ok(cached)
    }

    async fn resolve_github_access_token(
        &self,
        allow_device_login: bool,
    ) -> anyhow::Result<String> {
        if let Some(override_token) = self
            .github_token_override
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        {
            return Ok(override_token.to_string());
        }

        if let Some(stored) = self.store.load_github_token().await? {
            return Ok(stored);
        }

        if !allow_device_login {
            anyhow::bail!(
                "no cached GitHub token available; run `coproxy auth login` interactively"
            );
        }

        let token = device_flow::login_with_device_flow(&self.client).await?;
        self.store.save_github_token(token.as_str()).await?;
        Ok(token)
    }

    async fn exchange_github_for_ghcp(
        &self,
        github_access_token: &str,
    ) -> anyhow::Result<GhcpApiKeyResponse> {
        let mut request = self.client.get(GITHUB_API_KEY_URL);
        for (header, value) in copilot_headers() {
            request = request.header(header, value);
        }
        request = request.header("Authorization", format!("token {github_access_token}"));

        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                self.store.delete_github_token().await.ok();
            }
            anyhow::bail!(
                "failed to exchange GitHub token for GHCP token ({status}): {}",
                sanitize_error_body(&body)
            );
        }

        let parsed = response.json::<GhcpApiKeyResponse>().await?;
        Ok(parsed)
    }

    async fn chat_once(
        &self,
        request: &CreateChatCompletionRequest,
        model: &str,
        creds: &CachedToken,
    ) -> Result<ProviderChatResponse, ProviderError> {
        let mut upstream_request = (*request).clone();
        upstream_request.model = Some(model.to_string());
        upstream_request.stream = Some(false);
        if upstream_request.temperature.is_none() {
            upstream_request.temperature = Some(1.0);
        }

        let url = format!(
            "{}/chat/completions",
            creds.api_endpoint.trim_end_matches('/')
        );

        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", creds.token))
            .json(&upstream_request);

        for (header, value) in copilot_headers() {
            req = req.header(header, value);
        }

        let response = req.send().await.map_err(|error| {
            ProviderError::Upstream(format!("failed calling GHCP chat endpoint: {error}"))
        })?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED
            || response.status() == reqwest::StatusCode::FORBIDDEN
        {
            self.invalidate_ghcp_token_cache().await;
            return Err(ProviderError::Unauthorized(
                "GHCP token expired or invalid".to_string(),
            ));
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let message = format!(
                "GHCP chat completion failed ({status}): {}",
                sanitize_error_body(&body)
            );
            if status.is_client_error() {
                return Err(ProviderError::BadRequest(message));
            }
            return Err(ProviderError::Upstream(message));
        }

        let parsed = response
            .json::<UpstreamChatResponse>()
            .await
            .map_err(|error| {
                ProviderError::Upstream(format!("failed parsing GHCP response: {error}"))
            })?;

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Upstream("GHCP returned no choices".to_string()))?;

        let usage = parsed.usage.unwrap_or(UpstreamUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
        });
        Ok(ProviderChatResponse {
            model: parsed.model.unwrap_or_else(|| model.to_string()),
            content: choice.message.content,
            tool_calls: choice.message.tool_calls.unwrap_or_default(),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
        })
    }

    async fn stream_once(
        &self,
        request: &CreateChatCompletionRequest,
        creds: &CachedToken,
    ) -> Result<reqwest::Response, ProviderError> {
        let url = format!(
            "{}/chat/completions",
            creds.api_endpoint.trim_end_matches('/')
        );

        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", creds.token))
            .json(request);

        for (header, value) in copilot_headers() {
            req = req.header(header, value);
        }
        req = req.header("Accept", "text/event-stream");

        let response = req.send().await.map_err(|error| {
            ProviderError::Upstream(format!("failed calling GHCP chat endpoint: {error}"))
        })?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED
            || response.status() == reqwest::StatusCode::FORBIDDEN
        {
            self.invalidate_ghcp_token_cache().await;
            return Err(ProviderError::Unauthorized(
                "GHCP token expired or invalid".to_string(),
            ));
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let message = format!(
                "GHCP chat completion stream failed ({status}): {}",
                sanitize_error_body(&body)
            );
            if status.is_client_error() {
                return Err(ProviderError::BadRequest(message));
            }
            return Err(ProviderError::Upstream(message));
        }

        Ok(response)
    }

    async fn create_response_once(
        &self,
        request: &Value,
        creds: &CachedToken,
    ) -> Result<reqwest::Response, ProviderError> {
        let url = format!("{}/responses", creds.api_endpoint.trim_end_matches('/'));
        let mut req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", creds.token))
            .json(request);

        for (header, value) in copilot_headers() {
            req = req.header(header, value);
        }

        if request
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            req = req.header("Accept", "text/event-stream");
        }

        let response = req.send().await.map_err(|error| {
            ProviderError::Upstream(format!("failed calling GHCP responses endpoint: {error}"))
        })?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED
            || response.status() == reqwest::StatusCode::FORBIDDEN
        {
            self.invalidate_ghcp_token_cache().await;
            return Err(ProviderError::Unauthorized(
                "GHCP token expired or invalid".to_string(),
            ));
        }

        Ok(response)
    }

    async fn get_response_once(
        &self,
        response_id: &str,
        raw_query: Option<&str>,
        creds: &CachedToken,
    ) -> Result<reqwest::Response, ProviderError> {
        let url = build_upstream_url(&creds.api_endpoint, &["responses", response_id], raw_query)?;
        let mut req = self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", creds.token));

        for (header, value) in copilot_headers() {
            req = req.header(header, value);
        }

        let response = req.send().await.map_err(|error| {
            ProviderError::Upstream(format!("failed calling GHCP responses endpoint: {error}"))
        })?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED
            || response.status() == reqwest::StatusCode::FORBIDDEN
        {
            self.invalidate_ghcp_token_cache().await;
            return Err(ProviderError::Unauthorized(
                "GHCP token expired or invalid".to_string(),
            ));
        }

        Ok(response)
    }

    pub async fn create_response(
        &self,
        mut request: Value,
        default_model: Option<&str>,
    ) -> Result<reqwest::Response, ProviderError> {
        apply_default_model_to_response_request(&mut request, default_model);

        let creds = self
            .resolve_ghcp_token(false)
            .await
            .map_err(|error| ProviderError::Unauthorized(error.to_string()))?;

        match self.create_response_once(&request, &creds).await {
            Ok(response) => Ok(response),
            Err(ProviderError::Unauthorized(_)) => {
                let refreshed = self
                    .resolve_ghcp_token(false)
                    .await
                    .map_err(|error| ProviderError::Unauthorized(error.to_string()))?;
                self.create_response_once(&request, &refreshed).await
            }
            Err(other) => Err(other),
        }
    }

    pub async fn get_response(
        &self,
        response_id: &str,
        raw_query: Option<&str>,
    ) -> Result<reqwest::Response, ProviderError> {
        let creds = self
            .resolve_ghcp_token(false)
            .await
            .map_err(|error| ProviderError::Unauthorized(error.to_string()))?;

        match self.get_response_once(response_id, raw_query, &creds).await {
            Ok(response) => Ok(response),
            Err(ProviderError::Unauthorized(_)) => {
                let refreshed = self
                    .resolve_ghcp_token(false)
                    .await
                    .map_err(|error| ProviderError::Unauthorized(error.to_string()))?;
                self.get_response_once(response_id, raw_query, &refreshed)
                    .await
            }
            Err(other) => Err(other),
        }
    }

    pub async fn create_chat_completion_stream(
        &self,
        mut request: CreateChatCompletionRequest,
        default_model: Option<&str>,
    ) -> Result<reqwest::Response, ProviderError> {
        if request.messages.is_empty() {
            return Err(ProviderError::BadRequest(
                "`messages` must not be empty".to_string(),
            ));
        }

        request.model = Some(resolve_model(request.model.as_deref(), default_model));
        request.stream = Some(true);
        if request.temperature.is_none() {
            request.temperature = Some(1.0);
        }

        let creds = self
            .resolve_ghcp_token(false)
            .await
            .map_err(|error| ProviderError::Unauthorized(error.to_string()))?;

        match self.stream_once(&request, &creds).await {
            Ok(response) => Ok(response),
            Err(ProviderError::Unauthorized(_)) => {
                let refreshed = self
                    .resolve_ghcp_token(false)
                    .await
                    .map_err(|error| ProviderError::Unauthorized(error.to_string()))?;
                self.stream_once(&request, &refreshed).await
            }
            Err(other) => Err(other),
        }
    }

    async fn invalidate_ghcp_token_cache(&self) {
        {
            let mut lock = self.cached_ghcp_token.lock().await;
            *lock = None;
        }
        self.store.delete_ghcp_token().await.ok();
    }
}

impl ModelProvider for GhcpProvider {
    async fn create_chat_completion(
        &self,
        request: CreateChatCompletionRequest,
        default_model: Option<&str>,
    ) -> Result<ProviderChatResponse, ProviderError> {
        if request.messages.is_empty() {
            return Err(ProviderError::BadRequest(
                "`messages` must not be empty".to_string(),
            ));
        }

        if request.stream.unwrap_or(false) {
            return Err(ProviderError::NotSupported(
                "streaming is not implemented yet".to_string(),
            ));
        }

        let model = resolve_model(request.model.as_deref(), default_model);

        let creds = self
            .resolve_ghcp_token(false)
            .await
            .map_err(|error| ProviderError::Unauthorized(error.to_string()))?;

        match self.chat_once(&request, &model, &creds).await {
            Ok(response) => Ok(response),
            Err(ProviderError::Unauthorized(_)) => {
                let refreshed = self
                    .resolve_ghcp_token(false)
                    .await
                    .map_err(|error| ProviderError::Unauthorized(error.to_string()))?;
                self.chat_once(&request, &model, &refreshed).await
            }
            Err(other) => Err(other),
        }
    }
}

fn resolve_model(request_model: Option<&str>, default_model: Option<&str>) -> String {
    request_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            default_model
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

fn is_token_fresh(expires_at: i64) -> bool {
    chrono::Utc::now().timestamp() + 120 < expires_at
}

fn copilot_headers() -> [(&'static str, &'static str); 4] {
    [
        ("Editor-Version", "vscode/1.85.1"),
        ("Editor-Plugin-Version", "copilot/1.155.0"),
        ("User-Agent", "GithubCopilot/1.155.0"),
        ("Accept", "application/json"),
    ]
}

fn sanitize_error_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "empty body".to_string();
    }
    const MAX: usize = 400;
    if trimmed.len() <= MAX {
        return trimmed.to_string();
    }
    format!("{}...", &trimmed[..MAX])
}

fn apply_default_model_to_response_request(request: &mut Value, default_model: Option<&str>) {
    let Some(default_model) = default_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
    else {
        return;
    };

    let Some(payload) = request.as_object_mut() else {
        return;
    };

    let set_default = match payload.get("model") {
        None => true,
        Some(Value::Null) => true,
        Some(Value::String(existing)) => existing.trim().is_empty(),
        _ => false,
    };

    if set_default {
        payload.insert("model".to_string(), Value::String(default_model));
    }
}

fn build_upstream_url(
    base: &str,
    path_segments: &[&str],
    raw_query: Option<&str>,
) -> Result<reqwest::Url, ProviderError> {
    let mut url = reqwest::Url::parse(base).map_err(|error| {
        ProviderError::Internal(anyhow::anyhow!(
            "failed parsing upstream API endpoint URL: {error}"
        ))
    })?;

    {
        let mut segments = url.path_segments_mut().map_err(|_| {
            ProviderError::Internal(anyhow::anyhow!(
                "upstream API endpoint cannot be a base for path segments"
            ))
        })?;
        segments.pop_if_empty();
        for segment in path_segments {
            segments.push(segment);
        }
    }

    if let Some(query) = raw_query {
        if !query.trim().is_empty() {
            url.set_query(Some(query));
        }
    }

    Ok(url)
}

fn merge_with_default(mut models: Vec<String>, default_model: Option<&str>) -> Vec<String> {
    if let Some(custom) = default_model {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            models.push(trimmed.to_string());
        }
    }
    models.sort();
    models.dedup();
    models
}
