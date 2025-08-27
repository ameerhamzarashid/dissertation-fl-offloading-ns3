#!/usr/bin/env bash
set -euo pipefail
export BRIDGE_VARIANT=baseline
export NUM_PARAMS=${1:-2000000}
echo "BRIDGE_VARIANT=$BRIDGE_VARIANT NUM_PARAMS=$NUM_PARAMS"
