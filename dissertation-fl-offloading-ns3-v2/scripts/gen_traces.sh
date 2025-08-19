#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/traces
./ns3/build/mec_trace
echo "[ok] traces → experiments/traces/"
