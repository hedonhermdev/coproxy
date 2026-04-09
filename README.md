# coproxy

OpenAI-compatible API proxy backed by GitHub Copilot.

> [!WARNING]
> This is a reverse-engineered proxy of GitHub Copilot API. It is not supported by GitHub, and may break unexpectedly. Use at your own risk.

> [!WARNING]
> **GitHub Security Notice:**  
> Excessive automated or scripted use of Copilot (including rapid or bulk requests, such as via automated tools) may trigger GitHub's abuse-detection systems.  
> You may receive a warning from GitHub Security, and further anomalous activity could result in temporary suspension of your Copilot access.
>
> GitHub prohibits use of their servers for excessive automated bulk activity or any activity that places undue burden on their infrastructure.
>
> Please review:
>
> - [GitHub Acceptable Use Policies](https://docs.github.com/site-policy/acceptable-use-policies/github-acceptable-use-policies#4-spam-and-inauthentic-activity-on-github)
> - [GitHub Copilot Terms](https://docs.github.com/site-policy/github-terms/github-terms-for-additional-products-and-features#github-copilot)
>
> Use this proxy responsibly to avoid account restrictions.

## Installation

### From crates.io

```bash
cargo install coproxy
```

### From source

```bash
git clone https://github.com/hedonhermdev/coproxy.git
cd coproxy
cargo install --path .
```

### Prebuilt binaries

Download from [GitHub Releases](https://github.com/hedonhermdev/coproxy/releases). Available for Linux (x86_64, aarch64) and macOS (x86_64, aarch64).

### Docker

```bash
docker run -it -p 8080:8080 ghcr.io/hedonhermdev/coproxy serve --host 0.0.0.0
```

On first run, the container will print a GitHub device login URL and code to stdout. Complete the auth flow in your browser, and the server will start automatically.

To persist credentials across restarts, mount a volume:

```bash
docker run -it -p 8080:8080 -v coproxy-data:/data ghcr.io/hedonhermdev/coproxy serve --host 0.0.0.0 --state-dir /data
```

## Quick start

```bash
# First-time login (opens GitHub device flow)
coproxy auth login

# Start the proxy
coproxy serve --port 8080 --api-surface all

# Verify it works
curl http://127.0.0.1:8080/v1/models -H "Authorization: Bearer any-key"
```

## Usage with OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="any-key",  # or your --api-key value
)

# Chat completions
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Responses API (model-dependent, e.g. gpt-5.4)
response = client.responses.create(
    model="gpt-5.4",
    input="Explain quantum computing in one sentence.",
)
print(response.output_text)

# List models
for model in client.models.list():
    print(model.id)
```

## API surface

| Endpoint | Status |
|---|---|
| `POST /v1/chat/completions` | Supported (sync + stream) |
| `GET /v1/models` | Supported |
| `GET /v1/models/{model}` | Supported |
| `POST /v1/responses` | Passthrough (model-dependent) |
| `GET /v1/responses/{response_id}` | Passthrough |
| `POST /v1/embeddings` | Not supported |

## Configuration

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Bind port |
| `-d`, `--daemon` | false | Run the server as a background daemon |
| `--stop` | false | Stop a running daemon |
| `--api-surface` | `chat` | API surface: `chat`, `chat-responses`, `chat-embeddings`, `all` |
| `--api-key` | none | Require Bearer token for `/v1/*` routes |
| `--default-model` | `gpt-4o` | Default model when requests omit `model` |
| `--state-dir` | OS default | Override credential storage path |
| `--no-auto-login` | false | Skip automatic auth bootstrap at startup |
| `--log-level` | `info` | Log level filter |

### Environment variables

| Variable | Equivalent to |
|---|---|
| `GHCP_GITHUB_TOKEN` | `--github-token` (skip device flow, use this token directly) |
| `GHCP_PROXY_API_KEY` | `--api-key` |

## Authentication

On first run, coproxy triggers GitHub device flow:

1. Prints a verification URL and user code.
2. Polls GitHub OAuth until you authorize.
3. Exchanges GitHub token for a GHCP API token.
4. Caches both tokens locally with restricted file permissions (`0600`).

```bash
coproxy auth login    # Interactive login
coproxy auth status   # Show cached token info
coproxy auth logout   # Remove cached credentials
```

Or skip device flow entirely by providing a GitHub token:

```bash
export GHCP_GITHUB_TOKEN=ghp_xxxxx
coproxy serve
```

## Running as a daemon

Use `-d` to start the server in the background. The daemon's PID is written to `<state-dir>/coproxy.pid`.

```bash
# Start in background
coproxy serve -d --port 8080

# Stop the daemon
coproxy serve --stop
```

## Running as a service

### systemd (Linux)

Copy `contrib/coproxy.service` to `/etc/systemd/system/` and adjust as needed:

```bash
sudo cp contrib/coproxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now coproxy
```

### launchd (macOS)

Copy `contrib/com.coproxy.plist` to `~/Library/LaunchAgents/`:

```bash
cp contrib/com.coproxy.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.coproxy.plist
```

## Development

```bash
cargo fmt --all -- --check    # Format check
cargo clippy --all-targets    # Lint
cargo check                   # Type check
cargo test                    # Rust tests

# OpenAI compatibility tests (requires uv + GHCP auth)
scripts/run-openai-compat-tests.sh
```

## License

MIT
