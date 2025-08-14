#สคริปต์ช่วยวาง/อัปเดต token ง่าย ๆ: /workspace/set_token.sh
#ใช้เวลาคุณต้องอัปโหลด/อัปเดต token.json ใหม่ แล้วล็อกสิทธิ์ให้ถูกต้องอัตโนมัติ
#!/usr/bin/env bash
set -euo pipefail
CFG_DIR="/workspace/config"
mkdir -p "$CFG_DIR"

if [[ $# -eq 1 && -f "$1" ]]; then
  # กรณีระบุไฟล์ต้นทางในเครื่อง Pod อยู่แล้ว
  cp -f "$1" "$CFG_DIR/token.json"
elif [[ $# -eq 0 ]]; then
  # เปิด nano ให้แปะ token.json ด้วยมือ
  echo "[*] Opening editor. Paste your token.json, then save (Ctrl+O, Enter) and exit (Ctrl+X)."
  nano "$CFG_DIR/token.json"
else
  echo "Usage: $0  OR  $0 /path/to/token.json"
  exit 2
fi

chmod 600 "$CFG_DIR/token.json"
echo "[OK] token.json installed at $CFG_DIR/token.json with chmod 600"
