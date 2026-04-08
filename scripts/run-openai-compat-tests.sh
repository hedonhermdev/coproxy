#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

HOST="${GHCP_TEST_HOST:-127.0.0.1}"
PORT="${GHCP_TEST_PORT:-4010}"
API_SURFACE="${GHCP_TEST_API_SURFACE:-all}"
API_KEY="${TEST_API_KEY:-compat-test-key}"
BASE_URL="${TEST_API_BASE_URL:-http://${HOST}:${PORT}/v1}"

SERVER_LOG_DIR="$ROOT_DIR/.tmp"
SERVER_LOG_FILE="$SERVER_LOG_DIR/coproxy-compat-server.log"
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

echo "==> Creating virtual environment with uv"
uv venv --allow-existing "$VENV_DIR"

PYTHON_BIN="$VENV_DIR/bin/python"

echo "==> Installing Python test dependencies"
uv pip install --python "$PYTHON_BIN" -r "$ROOT_DIR/tests/requirements.txt"

echo "==> Starting coproxy compatibility server"
cargo run -- serve --host "$HOST" --port "$PORT" --api-surface "$API_SURFACE" --api-key "$API_KEY" --no-auto-login >"$SERVER_LOG_FILE" 2>&1 &
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

echo "==> Running compatibility tests"
export TEST_API_KEY="$API_KEY"
export TEST_API_BASE_URL="$BASE_URL"

"$PYTHON_BIN" -m pytest "$ROOT_DIR/tests" "$@"

echo "==> Compatibility tests completed"
