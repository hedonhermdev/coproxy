# OpenAI Compatibility Test Suite

This test package validates this server against the official `openai` Python client installed from PyPI via `uv`.

Current coverage targets the API surface implemented in this repo:

- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/chat/completions` (sync + stream)
- `POST /v1/responses` (sync + stream, model-dependent)
- `GET /v1/responses/{response_id}` passthrough behavior
- compatibility error contract for `POST /v1/embeddings`

## Quick Start

From repo root:

```bash
scripts/run-openai-compat-tests.sh
```

The script will:

1. Create/update `.venv` using `uv venv`.
2. Install dependencies using `uv pip`.
3. Start the cargo server on `127.0.0.1:4010`.
4. Run `pytest` for the compatibility suite.
5. Stop the server process on exit.

## Environment Variables

- `VENV_DIR` (default: `.venv`)
- `GHCP_TEST_HOST` (default: `127.0.0.1`)
- `GHCP_TEST_PORT` (default: `4010`)
- `GHCP_TEST_API_SURFACE` (default: `all`)
- `TEST_API_KEY` (default: `compat-test-key`)
- `TEST_API_BASE_URL` (default: `http://127.0.0.1:4010/v1`)
- `TEST_MODEL` (default: `gpt-4o`)
- `TEST_RESPONSES_MODEL` (default: `gpt-5.4`)

### GHCP Auth Note

Chat completion success tests require GHCP auth (cached token or `GHCP_GITHUB_TOKEN`).

Responses success tests additionally require a model that GHCP allows for the Responses API (`TEST_RESPONSES_MODEL`, default `gpt-5.4`).

If auth or model capability is unavailable, success-path tests are skipped and the rest of the suite still validates compatibility behavior and error formats.
