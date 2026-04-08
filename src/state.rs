use crate::provider::ghcp::GhcpProvider;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub provider: Arc<GhcpProvider>,
    pub api_key: Option<String>,
    pub default_model: Option<String>,
}

impl AppState {
    pub fn new(
        provider: GhcpProvider,
        api_key: Option<String>,
        default_model: Option<String>,
    ) -> Self {
        Self {
            provider: Arc::new(provider),
            api_key,
            default_model,
        }
    }
}
