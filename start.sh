#!/usr/bin/env bash
set -euo pipefail

LOG="/tmp/cloudflared.log"
OLLAMA_LOG="/tmp/ollama.log"
OLLAMA_PID=""

echo "[start] booting... $(date -u)"
APP_VERSION="$(python3 -c 'from version import __version__; print(__version__)')"
echo "[start] app version ${APP_VERSION}"
: "${CLOUDFLARE_TUNNEL_TOKEN:?CLOUDFLARE_TUNNEL_TOKEN is not set}"
command -v cloudflared >/dev/null || { echo "[start] cloudflared not found"; exit 1; }

# ----------------------------
# Blog/DB init (persistent on /workspace network volume)
# ----------------------------
mkdir -p /workspace/blog /workspace/uploads /workspace/unlabeled \
  /workspace/3d_outputs /workspace/tmp /workspace/ollama

# Defaults; can be overridden by RunPod env
export FLASK_APP="${FLASK_APP:-app.py}"
export DATABASE_URL="${DATABASE_URL:-sqlite:////workspace/blog/blog.db}"
export UPLOAD_DIR="${UPLOAD_DIR:-/workspace/uploads}"
export OUTPUT_2D="${OUTPUT_2D:-/workspace/unlabeled}"
export OUTPUT_3D="${OUTPUT_3D:-/workspace/3d_outputs}"
export WORKSPACE_TMP="${WORKSPACE_TMP:-/workspace/tmp}"
export WORKSPACE_BLOG_DIR="${WORKSPACE_BLOG_DIR:-/workspace/blog}"

# ----------------------------
# Optional local Ollama (coaching LLM)
# ENABLE_OLLAMA=1 (default in image). Set ENABLE_OLLAMA=0 and OPENAI_API_KEY=sk-...
# for cloud OpenAI instead (also clear/override OPENAI_BASE_URL).
# ----------------------------
ENABLE_OLLAMA="${ENABLE_OLLAMA:-1}"
if [[ "$ENABLE_OLLAMA" == "1" ]]; then
  export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
  export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/ollama}"
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:11434/v1}"
  export OPENAI_MODEL="${OPENAI_MODEL:-llama3.2:3b}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
  mkdir -p "$OLLAMA_MODELS"

  if command -v ollama >/dev/null 2>&1; then
    echo "[start] starting ollama on ${OLLAMA_HOST} (models=${OLLAMA_MODELS})..."
    ollama serve >"$OLLAMA_LOG" 2>&1 &
    OLLAMA_PID=$!
    for i in {1..60}; do
      if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo "[start] ollama is up"
        break
      fi
      if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "[start] ollama exited early; last log lines:"
        tail -n 40 "$OLLAMA_LOG" || true
        OLLAMA_PID=""
        break
      fi
      sleep 1
    done
    if [[ -n "$OLLAMA_PID" ]]; then
      echo "[start] ensuring model ${OPENAI_MODEL} is available (first boot may download)..."
      if ! ollama pull "${OPENAI_MODEL}"; then
        echo "[start] WARNING: ollama pull ${OPENAI_MODEL} failed; coaching may error until fixed"
        tail -n 40 "$OLLAMA_LOG" || true
      else
        echo "[start] model ${OPENAI_MODEL} ready"
      fi
    fi
  else
    echo "[start] WARNING: ENABLE_OLLAMA=1 but ollama binary missing"
  fi
else
  echo "[start] Ollama disabled (ENABLE_OLLAMA=0)"
fi

echo "[start] blog import..."
# If content/blog doesn't exist, blog-import prints a message and exits 0 (per our code)
flask blog-import

# ----------------------------
# Start app
# ----------------------------
echo "[start] starting gunicorn..."
gunicorn -k gthread -w "${WEB_CONCURRENCY:-1}" -b 0.0.0.0:5000 app:app &
GUNICORN_PID=$!

echo "[start] waiting for localhost:5000..."
for i in {1..30}; do
  if curl -fsS "http://127.0.0.1:5000/ping" >/dev/null 2>&1; then
    echo "[start] app is responding on /ping"
    break
  fi
  sleep 1
done

start_tunnel() {
  echo "[start] starting cloudflared tunnel..."
  # run in background, log to file
  cloudflared tunnel --no-autoupdate run \
  --token "${CLOUDFLARE_TUNNEL_TOKEN}" >"$LOG" 2>&1 &
  echo $!   # <-- ONLY the PID (no other output)
}

TUNNEL_PID="$(start_tunnel)"

echo "[start] waiting for tunnel to connect..."
for i in {1..60}; do
  if grep -qiE "Connected|Registered tunnel connection|Connection.*registered" "$LOG" 2>/dev/null; then
    echo "[start] tunnel connected"
    break
  fi

  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "[start] cloudflared exited early; showing last 60 log lines:"
    tail -n 60 "$LOG" || true
    echo "[start] restarting cloudflared..."
    TUNNEL_PID="$(start_tunnel)"
  fi
  sleep 1
done

term_handler() {
  echo "[start] shutting down..."
  kill "$TUNNEL_PID" 2>/dev/null || true
  kill "$GUNICORN_PID" 2>/dev/null || true
  if [[ -n "${OLLAMA_PID:-}" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
  fi
  wait || true
  exit 0
}
trap term_handler SIGTERM SIGINT

echo "[start] running. gunicorn=$GUNICORN_PID cloudflared=$TUNNEL_PID ollama=${OLLAMA_PID:-off}"
# Do not wait on ollama — if it dies, keep serving the app (coaching will error until restart).
wait -n "$GUNICORN_PID" "$TUNNEL_PID"
echo "[start] a process exited; dumping last 100 lines of cloudflared log:"
tail -n 100 "$LOG" || true
if [[ -f "$OLLAMA_LOG" ]]; then
  echo "[start] last 40 lines of ollama log:"
  tail -n 40 "$OLLAMA_LOG" || true
fi
exit 1
