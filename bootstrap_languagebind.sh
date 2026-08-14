#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
ENV_DIR=${LANGUAGEBIND_ENV_DIR:-$ROOT/venvs/languagebind}
REPO_DIR=${LANGUAGEBIND_REPO:-$ROOT/LanguageBind}
LOG=${LANGUAGEBIND_SETUP_LOG:-$ROOT/languagebind-setup.log}
PY=$(command -v python3 || command -v python)

echo "=== LanguageBind setup $(date -u) ===" | tee -a "$LOG"
mkdir -p "$(dirname "$ENV_DIR")"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$PY" -m venv --system-site-packages "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --no-cache-dir --upgrade pip >>"$LOG" 2>&1
"$ENV_DIR/bin/python" -m pip install --no-cache-dir \
  'transformers==4.30.2' sentencepiece einops timm ftfy regex >>"$LOG" 2>&1

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --depth 1 https://github.com/PKU-YuanGroup/LanguageBind.git "$REPO_DIR" >>"$LOG" 2>&1
else
  git -C "$REPO_DIR" pull --ff-only >>"$LOG" 2>&1 || true
fi

LANGUAGEBIND_REPO="$REPO_DIR" "$ENV_DIR/bin/python" - <<'PY' >>"$LOG" 2>&1
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
echo "=== LanguageBind setup complete ===" | tee -a "$LOG"
