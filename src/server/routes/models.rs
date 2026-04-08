use crate::openai::error::ApiError;
use crate::openai::types::{ListModelsResponse, Model};
use crate::server::routes::auth;
use crate::state::AppState;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::HeaderMap;

pub async fn list_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<ListModelsResponse>, ApiError> {
    auth::authorize(&headers, state.api_key.as_deref())?;
    let model_ids = state
        .provider
        .list_available_models(state.default_model.as_deref())
        .await
        .map_err(ApiError::from_provider_error)?;

    let now = chrono::Utc::now().timestamp();
    let data = model_ids
        .into_iter()
        .map(|id| Model {
            id,
            object: "model",
            created: now,
            owned_by: "github-copilot",
        })
        .collect();
    Ok(Json(ListModelsResponse {
        object: "list",
        data,
    }))
}

pub async fn get_model(
    Path(model): Path<String>,
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Model>, ApiError> {
    auth::authorize(&headers, state.api_key.as_deref())?;

    let known = state
        .provider
        .list_available_models(state.default_model.as_deref())
        .await
        .map_err(ApiError::from_provider_error)?;
    if !known.iter().any(|item| item == &model) {
        return Err(ApiError::not_found(format!("model not found: {model}")));
    }

    Ok(Json(Model {
        id: model,
        object: "model",
        created: chrono::Utc::now().timestamp(),
        owned_by: "github-copilot",
    }))
}
