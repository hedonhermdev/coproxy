mod routes;

use crate::cli::ApiSurface;
use crate::provider::ghcp::GhcpProvider;
use crate::state::AppState;
use axum::Router;
use axum::http::{HeaderName, Request, Response};
use std::time::Duration;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::{DefaultOnFailure, TraceLayer};
use tracing::{Level, Span, field, info};

#[derive(Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_surface: ApiSurface,
    pub api_key: Option<String>,
    pub default_model: Option<String>,
    /// When true, expose POST /v1/messages (Anthropic Messages API).
    pub anthropic_enabled: bool,
}

pub async fn run(config: ServerConfig, provider: GhcpProvider) -> anyhow::Result<()> {
    let state = AppState::new(provider, config.api_key, config.default_model);
    let app = app_router(config.api_surface, config.anthropic_enabled, state);

    let bind_addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&bind_addr).await?;
    let local_addr = listener.local_addr()?;
    let api_label = if config.anthropic_enabled {
        "Anthropic"
    } else {
        "OpenAI-compatible"
    };
    info!(
        "GHCP {} server listening on http://{}",
        api_label, local_addr
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn app_router(api_surface: ApiSurface, anthropic_enabled: bool, state: AppState) -> Router {
    let mut app = Router::new()
        .route("/healthz", axum::routing::get(routes::health::healthz))
        .route(
            "/v1/chat/completions",
            axum::routing::post(routes::chat_completions::create_chat_completion),
        )
        .route(
            "/v1/models",
            axum::routing::get(routes::models::list_models),
        )
        .route(
            "/v1/models/:model",
            axum::routing::get(routes::models::get_model),
        );

    if api_surface.responses_enabled() {
        app = app
            .route(
                "/v1/responses",
                axum::routing::post(routes::responses::create_response),
            )
            .route(
                "/v1/responses/:response_id",
                axum::routing::get(routes::responses::get_response),
            );
    }

    if api_surface.embeddings_enabled() {
        app = app.route(
            "/v1/embeddings",
            axum::routing::post(routes::embeddings::create_embeddings),
        );
    }

    if anthropic_enabled {
        app = app.route(
            "/v1/messages",
            axum::routing::post(routes::messages::create_message),
        );
    }

    let request_id_header = HeaderName::from_static("x-request-id");
    let middleware = ServiceBuilder::new()
        .layer(SetRequestIdLayer::new(
            request_id_header.clone(),
            MakeRequestUuid,
        ))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|req: &Request<_>| {
                    let request_id = req
                        .headers()
                        .get("x-request-id")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("-");
                    tracing::info_span!(
                        "http_request",
                        method = %req.method(),
                        path = %req.uri().path(),
                        request_id = %request_id,
                        status = field::Empty,
                        latency_ms = field::Empty,
                        model = field::Empty,
                        stream = field::Empty,
                    )
                })
                .on_request(|_: &Request<_>, _: &Span| {})
                .on_response(|resp: &Response<_>, latency: Duration, span: &Span| {
                    span.record("status", resp.status().as_u16());
                    span.record("latency_ms", latency.as_millis() as u64);
                    tracing::info!(target: "coproxy::access", "request complete");
                })
                .on_failure(DefaultOnFailure::new().level(Level::ERROR)),
        )
        .layer(PropagateRequestIdLayer::new(request_id_header));

    app.layer(middleware).with_state(state)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.ok();
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
