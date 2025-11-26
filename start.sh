#!/usr/bin/env bash
set -euo pipefail

# start.sh — starts backend (uvicorn) and frontend (static HTTP) in background
# Usage:
#   ./start.sh            # uses BACKEND_PORT=8000 FRONTEND_PORT=8080
#   FRONTEND_PORT=8080 ./start.sh

BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-8080}

export FRONTEND_PORT

# resolve script dir and use absolute log paths so tee can write even when child subshell cd's
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# If a .env file exists in the project root, source it and export its variables
# so child processes (uvicorn, python) inherit them. We use `set -a` to export
# any variables defined in the file, then disable automatic export afterwards.
if [ -f "$SCRIPT_DIR/.env" ]; then
  echo "Loading environment from $SCRIPT_DIR/.env"
  set -a
  # shellcheck disable=SC1090
  . "$SCRIPT_DIR/.env"
  set +a
fi

# Directory where incoming PPG payloads will be stored (default: project data folder)
# By default this points to the `data` folder next to the repository root so files
# are easy to find: $SCRIPT_DIR/data
export LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs}"
# Resolve absolute log directory so tee writes to a path independent of cwd
if [[ "$LOGDIR" = /* ]]; then
  ABS_LOGDIR="$LOGDIR"
else
  ABS_LOGDIR="$SCRIPT_DIR/${LOGDIR%/}"
fi
mkdir -p "$ABS_LOGDIR"
# Ensure the log files exist (create them if missing) so tee can append later.
touch "$ABS_LOGDIR/backend.log" "$ABS_LOGDIR/frontend.log" || true

# Determine PPG data directory and export it as an absolute path. If the
# environment variable `PPG_DATA_DIR` is provided and is absolute, use it.
# If it is provided and relative, interpret it relative to the repository
# root (`$SCRIPT_DIR`). If not provided, default to `$SCRIPT_DIR/data`.
if [ -z "${PPG_DATA_DIR+x}" ] || [ -z "$PPG_DATA_DIR" ]; then
  _ppg_dir="$SCRIPT_DIR/data"
else
  _ppg_dir="$PPG_DATA_DIR"
fi
case "$_ppg_dir" in
  /*) ABS_PPG_DATA_DIR="$_ppg_dir" ;;
  *) ABS_PPG_DATA_DIR="$SCRIPT_DIR/${_ppg_dir%/}" ;;
esac
# canonicalize (resolve symlinks) and export
ABS_PPG_DATA_DIR="$(cd "$ABS_PPG_DATA_DIR" && pwd)"
export PPG_DATA_DIR="$ABS_PPG_DATA_DIR"

# Determine PPG model path and export it as an absolute path. If the
# environment variable `PPG_MODEL_PATH` is provided and is absolute, use it.
# If it is provided and relative, interpret it relative to the repository
# root (`$SCRIPT_DIR`). If not provided, default to `$SCRIPT_DIR/models/model.keras`.
if [ -z "${PPG_MODEL_PATH+x}" ] || [ -z "$PPG_MODEL_PATH" ]; then
  _ppg_model="$SCRIPT_DIR/models/model.keras"
else
  _ppg_model="$PPG_MODEL_PATH"
fi
case "$_ppg_model" in
  /*) ABS_PPG_MODEL_PATH="$_ppg_model" ;;
  *) ABS_PPG_MODEL_PATH="$SCRIPT_DIR/${_ppg_model%/}" ;;
esac
# canonicalize (resolve symlinks) and export as a file path
ABS_PPG_MODEL_PATH="$(cd "$(dirname "$ABS_PPG_MODEL_PATH")" && pwd)/$(basename "$ABS_PPG_MODEL_PATH")"
export PPG_MODEL_PATH="$ABS_PPG_MODEL_PATH"

echo "Starting services..."

cd "$SCRIPT_DIR"

# Diagnostic info to help debug tee/file issues
echo "[debug] SCRIPT_DIR=$SCRIPT_DIR"
echo "[debug] LOGDIR=$LOGDIR"
echo "[debug] ABS_LOGDIR=$ABS_LOGDIR"
echo "[debug] PPG_DATA_DIR=$PPG_DATA_DIR"
echo "[debug] PPG_MODEL_PATH=$PPG_MODEL_PATH"
echo "[debug] PWD=$(pwd)"
if [ -d "$ABS_LOGDIR" ]; then
  echo "[debug] ABS_LOGDIR exists:" && ls -ld "$ABS_LOGDIR"
else
  echo "[debug] ABS_LOGDIR does NOT exist"
fi


# Start backend
if command -v uvicorn >/dev/null 2>&1; then
   echo "Starting backend (uvicorn) on port $BACKEND_PORT"
   # Pipe output through tee so logs appear both on console and in the log file
  # Use a shell loop to prefix lines so we don't depend on external tools like awk
  ( cd backend && FRONTEND_PORT="$FRONTEND_PORT" uvicorn main:app --host 0.0.0.0 --port "$BACKEND_PORT" 2>&1 | \
      while IFS= read -r line; do printf '[backend] %s\n' "$line"; done | tee "$ABS_LOGDIR/backend.log" ) &
  BACKEND_PID=$!
else
  echo "uvicorn not found in PATH. Please install it (pip install uvicorn) or activate your virtualenv." >&2
  exit 1
fi

# Start frontend static server
if command -v python >/dev/null 2>&1; then
   echo "Starting frontend (python -m http.server) on port $FRONTEND_PORT"
   # Pipe output through tee so logs appear both on console and in the log file
  ( cd frontend && python -m http.server "$FRONTEND_PORT" 2>&1 | \
      while IFS= read -r line; do printf '[frontend] %s\n' "$line"; done | tee "$ABS_LOGDIR/frontend.log" ) &
  FRONTEND_PID=$!
else
  echo "python not found in PATH. Please install Python 3." >&2
  kill "$BACKEND_PID" || true
  exit 1
fi

echo
echo "Backend: http://localhost:$BACKEND_PORT/"
echo "Frontend: http://localhost:$FRONTEND_PORT/index.html"
echo "Logs: $ABS_LOGDIR/backend.log  $ABS_LOGDIR/frontend.log"

cleanup() {
  echo "Stopping services..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
  wait "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}

trap cleanup INT TERM

# Wait for background processes
wait
