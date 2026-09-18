#!/usr/bin/env bash
# Dedicated SAM2 runtime. Never install its newer Torch into the media worker.
set -euo pipefail

ROOT=/workspace
WORKER_DIR=${WORKER_DIR:-/workspace/worker}
ENV_DIR=${SAM2_ENV_DIR:-$ROOT/venvs/sam2}
CHECKPOINT=${SAM2_CHECKPOINT:-$ROOT/checkpoints/sam2.1_hiera_large.pt}
PORT=${SAM2_PORT:-8101}
LOG=${SAM2_LOG:-$ROOT/sam2-service.log}
PY=$(command -v python3 || command -v python)

mkdir -p "$(dirname "$ENV_DIR")" "$(dirname "$CHECKPOINT")"
if [ ! -x "$ENV_DIR/bin/python" ]; then "$PY" -m venv "$ENV_DIR"; fi
"$ENV_DIR/bin/python" -m pip install --no-cache-dir --upgrade pip
"$ENV_DIR/bin/python" -m pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
SAM2_BUILD_CUDA=0 "$ENV_DIR/bin/python" -m pip install --no-cache-dir --no-deps --no-build-isolation git+https://github.com/facebookresearch/sam2.git
"$ENV_DIR/bin/python" -m pip install --no-cache-dir fastapi 'uvicorn[standard]' numpy pillow opencv-python-headless hydra-core iopath omegaconf tqdm
if [ ! -s "$CHECKPOINT" ]; then
  curl -fL --retry 3 -o "$CHECKPOINT.tmp" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
  mv "$CHECKPOINT.tmp" "$CHECKPOINT"
fi
pkill -f "uvicorn sam2_service:app" 2>/dev/null || true
cd "$WORKER_DIR"
LIBRARY_DIR="${LIBRARY_DIR:-$ROOT/library}" SAM2_CHECKPOINT="$CHECKPOINT" nohup "$ENV_DIR/bin/python" -m uvicorn sam2_service:app --host 127.0.0.1 --port "$PORT" >>"$LOG" 2>&1 &
sleep 3
curl -fsS "http://127.0.0.1:$PORT/health"
