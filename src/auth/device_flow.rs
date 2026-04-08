use anyhow::Context;
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

const GITHUB_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const GITHUB_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const GITHUB_ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default = "default_interval")]
    interval: u64,
    #[serde(default = "default_expires_in")]
    expires_in: u64,
}

fn default_interval() -> u64 {
    5
}

fn default_expires_in() -> u64 {
    900
}

#[derive(Debug, Deserialize)]
struct AccessTokenResponse {
    access_token: Option<String>,
    error: Option<String>,
}

pub async fn login_with_device_flow(client: &Client) -> anyhow::Result<String> {
    let response: DeviceCodeResponse = client
        .post(GITHUB_DEVICE_CODE_URL)
        .header("Accept", "application/json")
        .json(&serde_json::json!({
            "client_id": GITHUB_CLIENT_ID,
            "scope": "read:user"
        }))
        .send()
        .await
        .context("failed requesting GitHub device code")?
        .error_for_status()
        .context("GitHub rejected device code request")?
        .json()
        .await
        .context("failed parsing GitHub device code response")?;

    eprintln!(
        "\nGitHub authentication is required.\nVisit: {}\nCode: {}\nWaiting for authorization...\n",
        response.verification_uri, response.user_code
    );

    let mut poll_interval = Duration::from_secs(response.interval.max(5));
    let expires_at = tokio::time::Instant::now() + Duration::from_secs(response.expires_in.max(1));

    while tokio::time::Instant::now() < expires_at {
        tokio::time::sleep(poll_interval).await;

        let token_response: AccessTokenResponse = client
            .post(GITHUB_ACCESS_TOKEN_URL)
            .header("Accept", "application/json")
            .json(&serde_json::json!({
                "client_id": GITHUB_CLIENT_ID,
                "device_code": response.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
            }))
            .send()
            .await
            .context("failed polling GitHub access token")?
            .json()
            .await
            .context("failed parsing GitHub access token response")?;

        if let Some(token) = token_response.access_token {
            eprintln!("Authentication succeeded.\n");
            return Ok(token);
        }

        match token_response.error.as_deref() {
            Some("authorization_pending") | None => {}
            Some("slow_down") => {
                poll_interval += Duration::from_secs(5);
            }
            Some("expired_token") => anyhow::bail!("GitHub device authorization expired"),
            Some(other) => anyhow::bail!("GitHub device login failed: {other}"),
        }
    }

    anyhow::bail!("Timed out waiting for GitHub device authorization")
}
