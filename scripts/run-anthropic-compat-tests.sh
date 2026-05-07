#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="$ROOT_DIR/tests"

HOST="${GHCP_TEST_HOST:-127.0.0.1}"
PORT="${GHCP_TEST_ANTHROPIC_PORT:-4011}"
API_KEY="${TEST_API_KEY:-compat-test-key}"
BASE_URL="${TEST_ANTHROPIC_BASE_URL:-http://${HOST}:${PORT}}"

SERVER_LOG_DIR="$ROOT_DIR/.tmp"
SERVER_LOG_FILE="$SERVER_LOG_DIR/coproxy-anthropic-compat-server.log"
SERVER_PID=""

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing required command: $1" >&2
    exit 1
  fi
}

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

require_cmd uv
require_cmd cargo
require_cmd curl

mkdir -p "$SERVER_LOG_DIR"

echo "==> Verifying GHCP authentication"
if [ -n "${GHCP_GITHUB_TOKEN:-}" ]; then
  echo "    GHCP_GITHUB_TOKEN is set; skipping cached-token check"
elif cargo run --quiet -- auth status 2>/dev/null | grep -q "GitHub token cached: true"; then
  echo "    cached GitHub token found"
else
  echo "ERROR: no GHCP authentication available." >&2
  echo "       Either run 'coproxy auth login' to cache a token," >&2
  echo "       or set GHCP_GITHUB_TOKEN before invoking this script." >&2
  exit 1
fi

echo "==> Syncing test dependencies via uv (tests/pyproject.toml)"
uv sync --project "$TESTS_DIR"

echo "==> Starting coproxy compatibility server with --anthropic enabled"
cargo run -- serve --host "$HOST" --port "$PORT" --api-surface chat --anthropic --api-key "$API_KEY" --no-auto-login >"$SERVER_LOG_FILE" 2>&1 &
SERVER_PID="$!"

echo "==> Waiting for server readiness on http://$HOST:$PORT/healthz"
for _ in $(seq 1 120); do
  if curl --silent --fail "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl --silent --fail "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
  echo "ERROR: server failed to become ready" >&2
  echo "--- server log ---" >&2
  cat "$SERVER_LOG_FILE" >&2
  echo "------------------" >&2
  exit 1
fi

echo "==> Running Anthropic compatibility tests"
export TEST_API_KEY="$API_KEY"
export TEST_ANTHROPIC_BASE_URL="$BASE_URL"

uv run --project "$TESTS_DIR" pytest "$TESTS_DIR/anthropic" "$@"

echo "==> Anthropic compatibility tests completed"
