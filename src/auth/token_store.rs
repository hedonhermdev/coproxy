use anyhow::Context;
use chrono::{DateTime, Utc};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct TokenStore {
    root: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GhcpTokenRecord {
    pub token: String,
    pub expires_at: i64,
    pub api_endpoint: String,
}

#[derive(Debug)]
pub struct TokenStatus {
    pub github_token_cached: bool,
    pub ghcp_token_cached: bool,
    pub ghcp_expires_at: Option<DateTime<Utc>>,
}

impl TokenStore {
    pub fn new(state_dir_override: Option<PathBuf>) -> anyhow::Result<Self> {
        let root = state_dir_override
            .unwrap_or_else(default_state_dir)
            .join("auth");
        std::fs::create_dir_all(&root)
            .with_context(|| format!("failed creating state dir at {}", root.display()))?;
        set_dir_permissions(&root)?;
        Ok(Self { root })
    }

    pub fn root_dir(&self) -> &Path {
        &self.root
    }

    pub async fn load_github_token(&self) -> anyhow::Result<Option<String>> {
        read_text_file(self.github_token_path()).await
    }

    pub async fn save_github_token(&self, token: &str) -> anyhow::Result<()> {
        write_file_secure(self.github_token_path(), token).await
    }

    pub async fn delete_github_token(&self) -> anyhow::Result<()> {
        delete_if_exists(self.github_token_path()).await
    }

    pub async fn load_ghcp_token(&self) -> anyhow::Result<Option<GhcpTokenRecord>> {
        let Some(raw) = read_text_file(self.ghcp_token_path()).await? else {
            return Ok(None);
        };
        let parsed = serde_json::from_str::<GhcpTokenRecord>(&raw)
            .with_context(|| "failed parsing cached GHCP token JSON")?;
        Ok(Some(parsed))
    }

    pub async fn save_ghcp_token(&self, token: &GhcpTokenRecord) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(token)?;
        write_file_secure(self.ghcp_token_path(), &json).await
    }

    pub async fn delete_ghcp_token(&self) -> anyhow::Result<()> {
        delete_if_exists(self.ghcp_token_path()).await
    }

    pub async fn clear_all(&self) -> anyhow::Result<()> {
        self.delete_ghcp_token().await?;
        self.delete_github_token().await?;
        Ok(())
    }

    pub async fn status(&self) -> anyhow::Result<TokenStatus> {
        let github = self.load_github_token().await?.is_some();
        let ghcp = self.load_ghcp_token().await?;
        let ghcp_expires_at = ghcp
            .as_ref()
            .and_then(|token| DateTime::<Utc>::from_timestamp(token.expires_at, 0));
        Ok(TokenStatus {
            github_token_cached: github,
            ghcp_token_cached: ghcp.is_some(),
            ghcp_expires_at,
        })
    }

    fn github_token_path(&self) -> PathBuf {
        self.root.join("github-access-token")
    }

    fn ghcp_token_path(&self) -> PathBuf {
        self.root.join("ghcp-token.json")
    }
}

fn default_state_dir() -> PathBuf {
    ProjectDirs::from("", "", "coproxy")
        .map(|dirs| {
            dirs.state_dir()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| dirs.config_dir().to_path_buf())
        })
        .unwrap_or_else(|| {
            let user = std::env::var("USER")
                .or_else(|_| std::env::var("USERNAME"))
                .unwrap_or_else(|_| "unknown".to_string());
            std::env::temp_dir().join(format!("coproxy-{user}"))
        })
}

#[cfg(unix)]
fn set_dir_permissions(path: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("failed setting permissions on {}", path.display()))
}

#[cfg(not(unix))]
fn set_dir_permissions(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

async fn read_text_file(path: PathBuf) -> anyhow::Result<Option<String>> {
    let raw = match tokio::fs::read_to_string(&path).await {
        Ok(data) => data,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| format!("failed reading {}", path.display()));
        }
    };

    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(trimmed.to_string()))
}

async fn delete_if_exists(path: PathBuf) -> anyhow::Result<()> {
    match tokio::fs::remove_file(&path).await {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("failed deleting {}", path.display())),
    }
}

async fn write_file_secure(path: PathBuf, content: &str) -> anyhow::Result<()> {
    let content = content.to_string();
    tokio::task::spawn_blocking(move || {
        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&path)?;
            file.write_all(content.as_bytes())?;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
            Ok::<(), std::io::Error>(())
        }
        #[cfg(not(unix))]
        {
            std::fs::write(&path, content)?;
            Ok::<(), std::io::Error>(())
        }
    })
    .await
    .context("failed joining secure write task")??;
    Ok(())
}
