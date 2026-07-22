#!/usr/bin/env bash
# Thin wrapper — harvest is implemented in metal2vulkan-harvest.py.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
exec python3 "$DIR/metal2vulkan-harvest.py" "$@"
