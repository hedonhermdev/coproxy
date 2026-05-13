use clap::Parser;
use coproxy::auth::token_store::TokenStore;
use coproxy::cli::{ApiSurface, AuthCommand, ClaudeArgs, Cli, Command};
use coproxy::provider::ghcp::{GhcpProvider, ModelDetails};
use coproxy::server::{ServerConfig, run, serve_on_listener};
use std::io::IsTerminal;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // `coproxy claude` shares stderr with the child `claude` TUI; any log
    // output corrupts the terminal. Skip subscriber init entirely in that
    // path — anyhow errors still bubble up through main's return value.
    if !matches!(cli.command, Command::Claude(_)) {
        let env_filter = EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(cli.log_level.clone()))
            .unwrap_or_else(|_| EnvFilter::new("info"));
        let stderr_is_tty = std::io::stderr().is_terminal();
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_target(true)
            .with_ansi(stderr_is_tty)
            .with_writer(std::io::stderr)
            .init();
    }

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
        Command::Claude(args) => {
            run_claude(args, store, cli.github_token).await?;
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
                anthropic_enabled: args.anthropic,
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

    let mut cmd = Cmd::new(&exe);
    cmd.args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // Detach the child from the caller's console so closing the shell window
    // doesn't cascade a CTRL_CLOSE_EVENT into the daemon.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        cmd.creation_flags(DETACHED_PROCESS);
    }

    let child = cmd
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

/// Read the PID from `<state-dir>/coproxy.pid`, terminate the daemon, and remove the PID file.
///
/// Unix: sends SIGTERM so the server's `shutdown_signal` handler drains gracefully.
/// Windows: invokes `taskkill /F /PID <pid>` (hard kill). The proxy holds no durable
/// state across requests, so in-flight requests being dropped is acceptable.
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

        fs::remove_file(&pid_path).ok();
        println!("Sent SIGTERM to daemon (pid {pid})");
        Ok(())
    }

    #[cfg(windows)]
    {
        use std::process::{Command as Cmd, Stdio};

        let output = Cmd::new("taskkill")
            .args(["/F", "/PID", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .context("failed to invoke taskkill")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // taskkill exits 128 with "not found" / "not running" when the PID is gone.
            let not_found = stderr.contains("not found") || stderr.contains("not running");
            if not_found {
                fs::remove_file(&pid_path).ok();
                anyhow::bail!("process {pid} not found (stale PID file removed)");
            }
            anyhow::bail!("taskkill failed for pid {pid}: {}", stderr.trim());
        }

        fs::remove_file(&pid_path).ok();
        println!("Terminated daemon (pid {pid})");
        Ok(())
    }

    #[cfg(not(any(unix, windows)))]
    {
        anyhow::bail!("--stop is not supported on this platform (pid {pid})");
    }
}

use anyhow::Context;

/// Launch `claude` with an in-process Anthropic-compatible proxy.
///
/// Spins up the proxy as a tokio task (so it serves /v1/messages alongside),
/// then runs `claude` as a child process with ANTHROPIC_BASE_URL and
/// ANTHROPIC_API_KEY pointed at it. When `claude` exits, the proxy is shut
/// down and the process exits with claude's status code.
async fn run_claude(
    args: ClaudeArgs,
    store: TokenStore,
    github_token: Option<String>,
) -> anyhow::Result<()> {
    use std::process::Stdio;
    use tokio::process::Command as TokioCmd;

    let provider = GhcpProvider::new(store, github_token);
    let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    provider.ensure_ready(interactive).await?;

    // The proxy is loopback-only and lives for the duration of one child
    // process, so the api_key is just a sanity-check token — generate one
    // internally instead of asking the user for it.
    let api_key = format!("coproxy-{}", uuid::Uuid::new_v4());

    let cfg = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        api_surface: ApiSurface::Chat,
        api_key: Some(api_key.clone()),
        default_model: None,
        anthropic_enabled: true,
    };

    // Bind loopback on an OS-assigned port — the proxy is for the child
    // process we're about to spawn, nothing else needs to reach it.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to bind 127.0.0.1:0")?;
    let local_addr = listener.local_addr()?;
    let base_url = format!("http://{}", local_addr);

    // Start the server in the background on the pre-bound listener.
    let server_task = tokio::spawn(async move { serve_on_listener(cfg, provider, listener).await });

    let status = TokioCmd::new("claude")
        .args(&args.claude_args)
        .env("ANTHROPIC_BASE_URL", &base_url)
        .env("ANTHROPIC_API_KEY", &api_key)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("failed to spawn `claude` — is it installed and on PATH?")?;

    // Tear the server down. abort() is sufficient since the proxy holds no
    // durable state and any in-flight requests came from the child we just
    // reaped.
    server_task.abort();
    match server_task.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e.context("proxy server exited with error")),
        Err(join_err) if join_err.is_cancelled() => {}
        Err(join_err) => return Err(anyhow::anyhow!("proxy server task panicked: {join_err}")),
    }

    std::process::exit(status.code().unwrap_or(1));
}

fn pid_file_path(store: &TokenStore) -> std::path::PathBuf {
    store
        .root_dir()
        .parent()
        .map(|p| p.join("coproxy.pid"))
        .unwrap_or_else(|| store.root_dir().join("coproxy.pid"))
}

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
