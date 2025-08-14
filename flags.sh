#!/usr/bin/env bash
set -euo pipefail
CTRL_DIR="/workspace/.agent"
mkdir -p "$CTRL_DIR"
case "${1:-}" in
  enable-startup)  rm -f "$CTRL_DIR/DISABLE_STARTUP"; echo "Startup ENABLED";;
  disable-startup) touch "$CTRL_DIR/DISABLE_STARTUP"; echo "Startup DISABLED";;
  cancel-on)       touch "$CTRL_DIR/CANCEL"; echo "CANCEL set";;
  cancel-off)      rm -f "$CTRL_DIR/CANCEL"; echo "CANCEL cleared";;
  no-shutdown-on)  touch "$CTRL_DIR/NO_SHUTDOWN"; echo "NO_SHUTDOWN set";;
  no-shutdown-off) rm -f "$CTRL_DIR/NO_SHUTDOWN"; echo "NO_SHUTDOWN cleared";;
  status)          ls -la "$CTRL_DIR" || true;;
  *) echo "Usage: $0 {enable-startup|disable-startup|cancel-on|cancel-off|no-shutdown-on|no-shutdown-off|status}"; exit 2;;
esac
