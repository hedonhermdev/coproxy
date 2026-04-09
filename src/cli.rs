use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "coproxy")]
#[command(version)]
#[command(about = "Serve GHCP through an OpenAI-compatible API")]
pub struct Cli {
    #[arg(long, global = true)]
    pub state_dir: Option<PathBuf>,

    #[arg(long, global = true, env = "GHCP_GITHUB_TOKEN")]
    pub github_token: Option<String>,

    #[arg(long, global = true, default_value = "info")]
    pub log_level: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run OpenAI-compatible HTTP API server
    Serve(ServeArgs),
    /// Manage local GHCP authentication
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
    /// List available models
    Models {
        /// Output as JSON array
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ApiSurface {
    /// Serve /v1/chat/completions and /v1/models
    Chat,
    /// Also serve /v1/responses compatibility endpoints
    ChatResponses,
    /// Also serve /v1/embeddings endpoint
    ChatEmbeddings,
    /// Serve chat + responses + embeddings
    All,
}

impl ApiSurface {
    pub fn responses_enabled(self) -> bool {
        matches!(self, Self::ChatResponses | Self::All)
    }

    pub fn embeddings_enabled(self) -> bool {
        matches!(self, Self::ChatEmbeddings | Self::All)
    }
}

#[derive(Debug, Args)]
pub struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// API surface to expose from this binary
    #[arg(long, value_enum, default_value_t = ApiSurface::Chat)]
    pub api_surface: ApiSurface,

    /// Optional local API key to protect this proxy
    #[arg(long, env = "GHCP_PROXY_API_KEY")]
    pub api_key: Option<String>,

    /// Default model if requests omit `model`
    #[arg(long)]
    pub default_model: Option<String>,

    /// Skip automatic first-run login check
    #[arg(long)]
    pub no_auto_login: bool,

    /// Run the server as a background daemon
    #[arg(short = 'd', long, conflicts_with = "stop")]
    pub daemon: bool,

    /// Stop a running daemon
    #[arg(long)]
    pub stop: bool,
}

#[derive(Debug, Subcommand)]
pub enum AuthCommand {
    /// Login via GitHub token or device flow
    Login,
    /// Show auth status
    Status,
    /// Remove cached credentials
    Logout,
}
