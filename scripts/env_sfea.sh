#!/usr/bin/env bash
set -euo pipefail
export BRIDGE_VARIANT=sfea
export K_PERCENT=${1:-5}
export NUM_PARAMS=${2:-2000000}
echo "BRIDGE_VARIANT=$BRIDGE_VARIANT K_PERCENT=$K_PERCENT NUM_PARAMS=$NUM_PARAMS"
