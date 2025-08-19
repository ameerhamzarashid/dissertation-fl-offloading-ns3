#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."/ns3
rm -rf build && mkdir build && cd build
cmake .. -DNS3_ENABLE_TESTS=OFF -DNS3_ENABLE_EXAMPLES=OFF
cmake --build . -j"$(nproc)"
echo "[ok] built: ns3/build/mec_trace and ns3/build/mec_client"
