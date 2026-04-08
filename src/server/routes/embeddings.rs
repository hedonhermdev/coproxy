use crate::openai::error::ApiError;
use crate::server::routes::auth;
use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;

pub async fn create_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(_request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    auth::authorize(&headers, state.api_key.as_deref())?;
    Err(ApiError::not_supported(
        "embeddings are not implemented for GHCP provider",
    ))
}
