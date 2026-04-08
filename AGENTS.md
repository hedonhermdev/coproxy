# AGENTS.md

## Repo layout (actual entrypoints)
- Single-crate Rust project (`Cargo.toml`) with CLI + HTTP server.
- Main runtime entrypoints: `src/main.rs` (CLI command dispatch), `src/server/mod.rs` (Axum router + API-surface gating), `src/provider/ghcp.rs` (GHCP upstream calls, token exchange/cache, model catalog logic).
- OpenAI request/response wire types are defined in `src/openai/types.rs`; API error mapping is in `src/openai/error.rs`.

## Commands (no task runner wrappers)
- Start server: `cargo run -- serve --host 127.0.0.1 --port 8080`
- Auth lifecycle: `cargo run -- auth login`, `cargo run -- auth status`, `cargo run -- auth logout`
- Model listing CLI: `cargo run -- models --json`
- Focused verification: `cargo fmt --all -- --check`, `cargo clippy --all-targets`, `cargo test`, `cargo check`
- OpenAI compatibility tests: `scripts/run-openai-compat-tests.sh` (requires `uv` and GHCP auth)
- Current state: there are no Rust tests yet (`cargo test` runs 0 tests). Python compatibility tests live in `tests/`.

## Auth and state-dir gotchas
- By default, `serve` runs an auth readiness check at startup. On first run it can trigger GitHub device flow only when attached to a TTY.
- In non-interactive contexts, prefer `--no-auto-login`; otherwise startup can fail when no cached GitHub token exists.
- Global GitHub token override env: `GHCP_GITHUB_TOKEN`.
- Optional local proxy API key env: `GHCP_PROXY_API_KEY` (equivalent to `--api-key`).
- Token files are stored under `<state-dir>/auth/` as `github-access-token` and `ghcp-token.json` with restricted permissions on Unix (dir `0700`, files `0600`).

## Route-surface behavior (easy to misread)
- `--api-surface` controls which routes are mounted:
  - `chat` (default): `/v1/chat/completions`, `/v1/models`, `/v1/models/:model`
  - `chat-responses`: adds `/v1/responses` + `/v1/responses/:response_id` (passthrough to GHCP upstream)
  - `chat-embeddings`: adds `/v1/embeddings` (currently returns `not_supported`)
  - `all`: mounts all of the above
- `/healthz` is always exposed and does not use API-key auth.
- `/v1/*` routes enforce `Authorization: Bearer <key>` only when `--api-key`/`GHCP_PROXY_API_KEY` is configured.
- Streaming chat (`"stream": true`) is implemented via `create_chat_completion_stream` in the route layer; non-stream provider method intentionally rejects `stream=true`.
- `/v1/models` can still return a static fallback catalog when upstream GHCP model fetch/auth fails; chat completions do not have this fallback.
- `/v1/responses` proxies request/response directly to GHCP upstream; support is model-dependent.

## OpenAPI reference file
- `openapi.with-code-samples.yml` is a checked-in reference copy of the OpenAI API spec used to guide this repo's API compatibility work.
- It is not currently consumed by crate code, build scripts, or CLI wiring (no runtime/codegen dependency).
