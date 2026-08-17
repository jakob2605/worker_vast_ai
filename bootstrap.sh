#!/usr/bin/env bash
# Runs as the instance on-start command. Idempotent: safe to re-run.
#
# The dashboard base64-encodes this and passes it via onstart, because the API
# caps onstart at ~4 KB.
set -euo pipefail

WORKER_DIR=/workspace/worker
LIBRARY_DIR=${LIBRARY_DIR:-/workspace/library}
PORT=${WORKER_PORT:-8100}
LOG=/workspace/worker.log

echo "=== bootstrap $(date -u) ===" | tee -a "$LOG"

# ffmpeg must exist for both cutting and probing.
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v rclone >/dev/null 2>&1; then
  echo "installing ffmpeg and rclone" | tee -a "$LOG"
  apt-get update -qq && apt-get install -y -qq ffmpeg rclone python3-venv >>"$LOG" 2>&1
fi

# ----- Configure rclone from environment variable -----
if [ -n "${RCLONE_CONFIG_B64:-}" ]; then
  echo "Setting up rclone config from environment variable" | tee -a "$LOG"
  mkdir -p ~/.config/rclone
  echo "$RCLONE_CONFIG_B64" | base64 -d > ~/.config/rclone/rclone.conf
  chmod 600 ~/.config/rclone/rclone.conf

  # Test the remote (same command you ran manually)
  if rclone lsd gdrive:VastAIProgram >/dev/null 2>&1; then
    echo "rclone remote 'gdrive' is ready and can access VastAIProgram" | tee -a "$LOG"
  else
    echo "WARNING: rclone remote test failed - check the config" | tee -a "$LOG"
  fi
else
  echo "No RCLONE_CONFIG_B64 provided - rclone remote not configured" | tee -a "$LOG"
fi

mkdir -p "$WORKER_DIR" "$LIBRARY_DIR"

# Preferred code delivery when SSH-based `vastai copy` is not an option:
# clone straight from a repo. Set WORKER_GIT_URL to enable.
if [ -n "${WORKER_GIT_URL:-}" ]; then
  echo "fetching worker from $WORKER_GIT_URL" | tee -a "$LOG"
  command -v git >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq git >>"$LOG" 2>&1)
  if [ -d "$WORKER_DIR/.git" ]; then
    git -C "$WORKER_DIR" pull --ff-only >>"$LOG" 2>&1 || true
  else
    rm -rf "$WORKER_DIR"
    git clone --depth 1 "$WORKER_GIT_URL" "$WORKER_DIR" >>"$LOG" 2>&1
  fi
  # Allow the worker to live in a subdirectory of the repo.
  if [ -n "${WORKER_SUBDIR:-}" ] && [ -d "$WORKER_DIR/$WORKER_SUBDIR" ]; then
    WORKER_DIR="$WORKER_DIR/$WORKER_SUBDIR"
  fi
fi

# Find worker.py wherever the upload actually put it. `vastai copy` of a folder
# into an existing folder nests it (/workspace/worker/worker/worker.py), so
# checking a single fixed path misses a perfectly good upload.
find_worker() {
  for candidate in \
      /workspace/worker/worker.py \
      /workspace/worker/worker/worker.py \
      /workspace/worker.py \
      /workspace/Vast_AI_Program/worker/worker.py; do
    if [ -f "$candidate" ]; then
      dirname "$candidate"
      return 0
    fi
  done
  # Last resort: search /workspace for it.
  found=$(find /workspace -maxdepth 4 -name worker.py -not -path '*/pipeline/*' 2>/dev/null | head -1)
  [ -n "$found" ] && { dirname "$found"; return 0; }
  return 1
}

# The code is usually uploaded AFTER the instance boots, so wait for it rather
# than failing. Upload from the dashboard and the worker starts by itself.
if ! FOUND_DIR=$(find_worker); then
  echo "worker.py not here yet - waiting up to 30 min for an upload" | tee -a "$LOG"
  for _ in $(seq 1 180); do
    if FOUND_DIR=$(find_worker); then break; fi
    sleep 10
  done
fi

if ! FOUND_DIR=$(find_worker); then
  echo "ERROR: no worker.py anywhere under /workspace after waiting." | tee -a "$LOG"
  echo "Contents of /workspace:" | tee -a "$LOG"
  ls -laR /workspace 2>/dev/null | head -60 | tee -a "$LOG"
  exit 1
fi

WORKER_DIR="$FOUND_DIR"
echo "worker.py found at $WORKER_DIR" | tee -a "$LOG"
ls -l "$WORKER_DIR" | tee -a "$LOG"

# Python deps. torch ships with the vastai/pytorch image; never reinstall it,
# a pip torch would likely be the CPU build and silently kill GPU throughput.
if [ -f "$WORKER_DIR/requirements.txt" ]; then
  echo "installing python deps (this takes a few minutes)" | tee -a "$LOG"
  REQ_NO_TRANSNET=/tmp/worker-requirements-no-transnet.txt
  grep -v -E '^(transnetv2-pytorch|torch)([<=> ].*)?$' "$WORKER_DIR/requirements.txt" > "$REQ_NO_TRANSNET"
  if pip install --no-cache-dir -r "$REQ_NO_TRANSNET" >>"$LOG" 2>&1 \
      && pip install --no-cache-dir --no-deps transnetv2-pytorch >>"$LOG" 2>&1; then
    echo "deps installed" | tee -a "$LOG"
  else
    # Do not swallow this. A failed install is the most likely reason the
    # worker never comes up.
    echo "PIP INSTALL FAILED - last 20 lines:" | tee -a "$LOG"
    tail -20 "$LOG"
  fi
fi

# Bare 'uvicorn' is not always on PATH in these images; the module form always is.
PY=$(command -v python3 || command -v python)
echo "python: $PY" | tee -a "$LOG"
"$PY" -c "import uvicorn, fastapi; print('uvicorn + fastapi importable')" 2>&1 | tee -a "$LOG" || {
  echo "FATAL: uvicorn/fastapi missing after install" | tee -a "$LOG"; exit 1;
}

"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | tee -a "$LOG" || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG" || true

if [ "${INSTALL_LANGUAGEBIND:-1}" != "0" ] && [ -x "$WORKER_DIR/bootstrap_languagebind.sh" ]; then
  if [ ! -x /workspace/venvs/languagebind/bin/python ]; then
    echo "starting isolated LanguageBind setup in background" | tee -a "$LOG"
    nohup bash "$WORKER_DIR/bootstrap_languagebind.sh" >>/workspace/languagebind-setup.log 2>&1 &
  fi
fi

if [ -n "${RESTORE_SNAPSHOT:-}" ]; then
  echo "restoring Google Drive snapshot ${RESTORE_SNAPSHOT}" | tee -a "$LOG"
  (cd "$WORKER_DIR" && "$PY" -m pipeline.cloud_backup restore "$RESTORE_SNAPSHOT") 2>&1 | tee -a "$LOG"
fi

pkill -f "uvicorn worker:app" 2>/dev/null || true
sleep 1

cd "$WORKER_DIR"
export LIBRARY_DIR
export WORKER_TOKEN="${WORKER_TOKEN:-}"
nohup "$PY" -m uvicorn worker:app --host 0.0.0.0 --port "$PORT" >>"$LOG" 2>&1 &

echo "worker starting on :$PORT (library $LIBRARY_DIR)" | tee -a "$LOG"
sleep 4
curl -s "http://127.0.0.1:$PORT/health" | tee -a "$LOG" || echo "health check not ready yet" | tee -a "$LOG"
echo "=== bootstrap done ===" | tee -a "$LOG"
