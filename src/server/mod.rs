mod routes;

use crate::cli::ApiSurface;
use crate::provider::ghcp::GhcpProvider;
use crate::state::AppState;
use axum::Router;
use tokio::net::TcpListener;
use tracing::info;

#[derive(Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_surface: ApiSurface,
    pub api_key: Option<String>,
    pub default_model: Option<String>,
}

pub async fn run(config: ServerConfig, provider: GhcpProvider) -> anyhow::Result<()> {
    let state = AppState::new(provider, config.api_key, config.default_model);
    let app = app_router(config.api_surface, state);

    let bind_addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&bind_addr).await?;
    let local_addr = listener.local_addr()?;
    info!(
        "GHCP OpenAI-compatible server listening on http://{}",
        local_addr
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn app_router(api_surface: ApiSurface, state: AppState) -> Router {
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

    app.with_state(state)
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
