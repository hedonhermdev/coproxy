# Compatibility Test Suites

Two parallel suites validate this server against the official client SDKs
installed from PyPI via `uv`:

- **OpenAI** (`tests/`, top-level): runs the `openai` Python client against
  `/v1/chat/completions`, `/v1/models`, `/v1/responses`, and the not-supported
  contract for `/v1/embeddings`.
- **Anthropic** (`tests/anthropic/`): runs the `anthropic` Python client
  against `/v1/messages` (sync + stream, system messages, tool use, error
  shapes). Only meaningful when the server is started with `--anthropic`.

## Quick Start

From repo root:

```bash
# OpenAI suite
scripts/run-openai-compat-tests.sh

# Anthropic suite
scripts/run-anthropic-compat-tests.sh
```

Each script will:

1. `uv sync --project tests` — installs deps from `tests/pyproject.toml` into `.venv` (created on first run).
2. Start the cargo server on a dedicated port.
3. `uv run --project tests pytest …` for that suite.
4. Stop the server process on exit.

The Anthropic runner adds `--anthropic` to the cargo invocation so
`/v1/messages` is exposed.

## Environment Variables

### Common

- `VENV_DIR` (default: `.venv`)
- `GHCP_TEST_HOST` (default: `127.0.0.1`)
- `TEST_API_KEY` (default: `compat-test-key`)

### OpenAI suite

- `GHCP_TEST_PORT` (default: `4010`)
- `GHCP_TEST_API_SURFACE` (default: `all`)
- `TEST_API_BASE_URL` (default: `http://127.0.0.1:4010/v1`)
- `TEST_MODEL` (default: `gpt-4o`)
- `TEST_RESPONSES_MODEL` (default: `gpt-5.4`)
- `TEST_RESPONSES_IMAGE_MODEL` (default: `gpt-5.5`) — vision-capable model used
  by the Responses API image-input test; the test skips if the model rejects
  the request.

### Anthropic suite

- `GHCP_TEST_ANTHROPIC_PORT` (default: `4011`)
- `TEST_ANTHROPIC_BASE_URL` (default: `http://127.0.0.1:4011`)
- `TEST_ANTHROPIC_MODEL` (default: `claude-haiku-4.5`)
- `TEST_ANTHROPIC_IMAGE_MODEL` (default: `claude-opus-4.7`) — vision-capable
  model used by the image-input test; the test skips if the model rejects the
  request.

### GHCP Auth Note

Chat / message creation tests require GHCP auth (cached token or
`GHCP_GITHUB_TOKEN`). Responses success tests additionally require a model that
GHCP allows for the Responses API (`TEST_RESPONSES_MODEL`, default `gpt-5.4`).

If auth or model capability is unavailable, success-path tests are skipped and
the rest of each suite still validates compatibility behavior and error
formats.
