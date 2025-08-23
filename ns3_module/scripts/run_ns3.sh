#!/usr/bin/env bash
set -euo pipefail

# First arg kept for compatibility (not used to select binary path)
BUILD_TYPE=${1:-Release}
shift || true

# Resolve repo root (two levels up from this script: ns3_module/scripts -> repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN_PATH="$ROOT_DIR/ns3_module/build/run_mec_scenario"

if [[ ! -x "$BIN_PATH" ]]; then
  echo "Binary not found at $BIN_PATH. Build first with ns3_module/scripts/build_ns3_module.sh"
  exit 1
fi

# Run from repo root so relative paths like data/raw_logs land in the expected location
cd "$ROOT_DIR"
"$BIN_PATH" "$@"
