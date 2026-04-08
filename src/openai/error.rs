use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::provider::ProviderError;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub error_type: &'static str,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            message: message.into(),
            param: None,
            code: None,
        }
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            error_type: "invalid_request_error",
            message: message.into(),
            param: None,
            code: Some("unauthorized".to_string()),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error_type: "invalid_request_error",
            message: message.into(),
            param: None,
            code: Some("not_found".to_string()),
        }
    }

    pub fn not_supported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            message: message.into(),
            param: None,
            code: Some("not_supported".to_string()),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: "server_error",
            message: message.into(),
            param: None,
            code: None,
        }
    }

    pub fn from_provider_error(error: ProviderError) -> Self {
        match error {
            ProviderError::BadRequest(msg) => Self::bad_request(msg),
            ProviderError::Unauthorized(msg) => Self::unauthorized(msg),
            ProviderError::NotFound(msg) => Self::not_found(msg),
            ProviderError::NotSupported(msg) => Self::not_supported(msg),
            ProviderError::Upstream(msg) => Self::internal(msg),
            ProviderError::Internal(err) => Self::internal(err.to_string()),
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorPayload,
}

#[derive(Debug, Serialize)]
struct ErrorPayload {
    message: String,
    #[serde(rename = "type")]
    kind: String,
    param: Option<String>,
    code: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let error = ErrorPayload {
            message: self.message,
            kind: self.error_type.to_string(),
            param: self.param,
            code: self.code,
        };

        let wrapped = ErrorResponse {
            error: ErrorPayload {
                message: error.message.clone(),
                kind: error.kind.clone(),
                param: error.param.clone(),
                code: error.code.clone(),
            },
        };

        let payload = serde_json::json!({
            "message": error.message,
            "type": error.kind,
            "param": error.param,
            "code": error.code,
            "error": wrapped.error,
        });

        (self.status, Json(payload)).into_response()
    }
}
