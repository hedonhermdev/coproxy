use clap::Parser;
use coproxy::auth::token_store::TokenStore;
use coproxy::cli::{AuthCommand, Cli, Command};
use coproxy::provider::ghcp::{GhcpProvider, ModelDetails};
use coproxy::server::{ServerConfig, run};
use std::io::IsTerminal;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(cli.log_level.clone()))
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let store = TokenStore::new(cli.state_dir.clone())?;

    match cli.command {
        Command::Auth { command } => {
            let provider = GhcpProvider::new(store.clone(), cli.github_token.clone());
            match command {
                AuthCommand::Login => {
                    provider.ensure_ready(true).await?;
                    println!("Login successful.");
                }
                AuthCommand::Status => {
                    let status = store.status().await?;
                    println!("State dir: {}", store.root_dir().display());
                    println!("GitHub token cached: {}", status.github_token_cached);
                    println!("GHCP token cached: {}", status.ghcp_token_cached);
                    if let Some(expires_at) = status.ghcp_expires_at {
                        println!("GHCP token expires at: {expires_at}");
                    }
                }
                AuthCommand::Logout => {
                    store.clear_all().await?;
                    println!("Logged out (local tokens removed).");
                }
            }
        }
        Command::Models { json, verbose } => {
            let provider = GhcpProvider::new(store, cli.github_token);

            if json {
                // JSON always carries full upstream details.
                let details = provider.list_model_details().await?;
                let values: Vec<serde_json::Value> = details
                    .into_iter()
                    .map(|d| serde_json::Value::Object(d.raw))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&values)?);
            } else if verbose {
                let details = provider.list_model_details().await?;
                for (idx, detail) in details.iter().enumerate() {
                    if idx > 0 {
                        println!();
                    }
                    print_model_details(detail);
                }
            } else {
                let models = provider.list_available_models(None).await?;
                for model in models {
                    println!("{model}");
                }
            }
        }
        Command::Serve(args) => {
            if args.host.trim().is_empty() {
                anyhow::bail!("--host must not be empty");
            }

            if args.daemon {
                return daemonize(&store);
            }

            if args.stop {
                return stop_daemon(&store);
            }

            let provider = GhcpProvider::new(store, cli.github_token);
            if !args.no_auto_login {
                let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
                provider.ensure_ready(interactive).await?;
            }

            let cfg = ServerConfig {
                host: args.host,
                port: args.port,
                api_surface: args.api_surface,
                api_key: args.api_key,
                default_model: args.default_model,
            };
            run(cfg, provider).await?;
        }
    }

    Ok(())
}

/// Re-exec the current binary with the same arguments but without `-d`/`--daemon`,
/// detach the child process, write its PID to `<state-dir>/coproxy.pid`, and exit.
fn daemonize(store: &TokenStore) -> anyhow::Result<()> {
    use std::fs;
    use std::process::{Command as Cmd, Stdio};

    let exe = std::env::current_exe().context("failed to determine current executable path")?;

    // Rebuild args, stripping -d / --daemon.
    let args: Vec<String> = std::env::args()
        .skip(1) // skip argv[0]
        .filter(|a| a != "-d" && a != "--daemon")
        .collect();

    let child = Cmd::new(&exe)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn daemon child process")?;

    let pid = child.id();

    let pid_path = pid_file_path(store);
    fs::write(&pid_path, pid.to_string())
        .with_context(|| format!("failed to write PID file at {}", pid_path.display()))?;

    println!("Server started in background (pid {pid})");
    println!("PID file: {}", pid_path.display());
    Ok(())
}

/// Read the PID from `<state-dir>/coproxy.pid`, send SIGTERM, and remove the PID file.
fn stop_daemon(store: &TokenStore) -> anyhow::Result<()> {
    use std::fs;

    let pid_path = pid_file_path(store);

    let raw = fs::read_to_string(&pid_path).with_context(|| {
        format!(
            "no PID file found at {} — is the daemon running?",
            pid_path.display()
        )
    })?;

    let pid: u32 = raw
        .trim()
        .parse()
        .with_context(|| format!("invalid PID in {}: {:?}", pid_path.display(), raw.trim()))?;

    #[cfg(unix)]
    {
        use std::io;
        // Send SIGTERM for graceful shutdown (matches the server's shutdown_signal handler).
        let ret = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if ret != 0 {
            let err = io::Error::last_os_error();
            // ESRCH means process doesn't exist — stale PID file, clean it up anyway.
            if err.raw_os_error() == Some(libc::ESRCH) {
                fs::remove_file(&pid_path).ok();
                anyhow::bail!("process {pid} not found (stale PID file removed)");
            }
            return Err(err).context(format!("failed to send SIGTERM to pid {pid}"));
        }
    }

    #[cfg(not(unix))]
    {
        anyhow::bail!("--stop is only supported on Unix platforms");
    }

    fs::remove_file(&pid_path).ok();
    println!("Sent SIGTERM to daemon (pid {pid})");
    Ok(())
}

fn pid_file_path(store: &TokenStore) -> std::path::PathBuf {
    store
        .root_dir()
        .parent()
        .map(|p| p.join("coproxy.pid"))
        .unwrap_or_else(|| store.root_dir().join("coproxy.pid"))
}

use anyhow::Context;

/// Render a single model's details to stdout in a human-readable block.
/// Prefers well-known GHCP fields (vendor, version, capabilities.limits,
/// capabilities.supports, tokenizer) and falls back to `key: <json>` for any
/// other top-level fields so nothing upstream goes hidden.
fn print_model_details(detail: &ModelDetails) {
    use serde_json::Value;

    let raw = &detail.raw;
    println!("{}", detail.id);

    // Surface a short list of commonly-useful top-level scalars first.
    const HEADLINE_KEYS: &[&str] = &["name", "vendor", "version", "object", "preview"];
    for key in HEADLINE_KEYS {
        if let Some(value) = raw.get(*key)
            && let Some(rendered) = render_scalar(value)
        {
            println!("  {key}: {rendered}");
        }
    }

    // Capabilities: nested object with `family`, `type`, `tokenizer`, `limits`, `supports`.
    if let Some(Value::Object(caps)) = raw.get("capabilities") {
        println!("  capabilities:");
        for key in ["family", "type", "tokenizer"] {
            if let Some(v) = caps.get(key).and_then(render_scalar) {
                println!("    {key}: {v}");
            }
        }
        if let Some(Value::Object(limits)) = caps.get("limits") {
            println!("    limits:");
            for (k, v) in limits {
                if let Some(rendered) = render_scalar(v) {
                    println!("      {k}: {rendered}");
                } else {
                    println!("      {k}: {v}");
                }
            }
        }
        if let Some(Value::Object(supports)) = caps.get("supports") {
            let enabled: Vec<&str> = supports
                .iter()
                .filter_map(|(k, v)| (v.as_bool() == Some(true)).then_some(k.as_str()))
                .collect();
            if !enabled.is_empty() {
                println!("    supports: {}", enabled.join(", "));
            }
        }
    }

    // Anything else at the top level that we did not explicitly render above.
    let handled: &[&str] = &[
        "id",
        "name",
        "vendor",
        "version",
        "object",
        "preview",
        "capabilities",
    ];
    for (key, value) in raw {
        if handled.contains(&key.as_str()) {
            continue;
        }
        match render_scalar(value) {
            Some(rendered) => println!("  {key}: {rendered}"),
            None => println!("  {key}: {value}"),
        }
    }
}

fn render_scalar(value: &serde_json::Value) -> Option<String> {
    use serde_json::Value;
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}
