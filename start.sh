#!/usr/bin/env bash
set -euo pipefail

# ===== PATH/CONFIG =====
APP_DIR=/workspace/ai-batch
COMFY_DIR=/workspace/ComfyUI
VENVDIR=/workspace/venv
OUT_DIR=/workspace/outputs
LOG_DIR=/workspace/logs
CFG_DIR=/workspace/config
CTRL_DIR=/workspace/.agent
MODEL_DIR=/workspace/models/checkpoints
PORT=${COMFY_PORT:-8188}
PROMPT_API_BASE_DEFAULT="https://manus-api-server.onrender.com"

mkdir -p "$APP_DIR" "$OUT_DIR" "$LOG_DIR" "$CFG_DIR" "$CTRL_DIR" "$MODEL_DIR"

# ถ้ามี flag ปิด startup ก็ไม่ต้องทำอะไร
if [[ -f "$CTRL_DIR/DISABLE_STARTUP" ]]; then
  echo "[SKIP] DISABLE_STARTUP present. Exit."
  exit 0
fi

echo "[1/7] Install apt packages"
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git curl wget python3-venv python3-pip exiftool tmux ca-certificates

echo "[2/7] Python venv"
if [[ ! -d "$VENVDIR" ]]; then
  python3 -m venv "$VENVDIR"
fi
source "$VENVDIR/bin/activate"
python -m pip install -U pip wheel

echo "[3/7] PyTorch (CUDA 12.1) + libs"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install requests google-api-python-client google-auth-httplib2 google-auth-oauthlib realesrgan

echo "[4/7] ComfyUI setup"
if [[ ! -d "$COMFY_DIR" ]]; then
  git clone --depth=1 https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
fi
pip install -r "$COMFY_DIR/requirements.txt"

echo "[5/7] Models (SD1.5)"
cd "$MODEL_DIR"
if [[ ! -f "v1-5-pruned-emaonly.safetensors" ]]; then
  wget -O v1-5-pruned-emaonly.safetensors \
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" || true
fi

echo "[6/7] Pull latest app scripts from GitHub"
# ต้องแก้ GITHUB_USER ให้เป็นของคุณเอง ก่อนเอา one-liner ไปใช้
#GITHUB_USER="${GITHUB_USER:-YOUR_GITHUB_USERNAME}"
GITHUB_USER="${GITHUB_USER:-siamsia}"
RAW="https://raw.githubusercontent.com/${GITHUB_USER}/ai-batch/main"
curl -fsSL "$RAW/automation.py" -o "$APP_DIR/automation.py"
curl -fsSL "$RAW/run_batch.sh"   -o /workspace/run_batch.sh
curl -fsSL "$RAW/flags.sh"       -o /workspace/flags.sh
chmod +x /workspace/run_batch.sh /workspace/flags.sh

echo "[7/7] Start ComfyUI (if not running) + run batch"
# start ComfyUI once in tmux session
if ! pgrep -f "ComfyUI/main.py" >/dev/null 2>&1; then
  tmux new-session -d -s comfy "cd $COMFY_DIR && \
    $VENVDIR/bin/python main.py --listen 127.0.0.1 --port $PORT \
    >> $LOG_DIR/comfy.log 2>&1"
fi

# ให้ PROMPT_API_BASE มีค่าเสมอ
export PROMPT_API_BASE="${PROMPT_API_BASE:-$PROMPT_API_BASE_DEFAULT}"

# รันงานหนึ่งรอบใน tmux (ดู log ได้)
tmux new-session -d -s batch "$VENVDIR/bin/python $APP_DIR/automation.py \
  >> $LOG_DIR/batch.log 2>&1"

echo "=== READY ===
- Logs: tail -f $LOG_DIR/batch.log
- Flags helper: /workspace/flags.sh {enable-startup|disable-startup|cancel-on|cancel-off|no-shutdown-on|no-shutdown-off|status}
- Run batch again: /workspace/run_batch.sh
"
