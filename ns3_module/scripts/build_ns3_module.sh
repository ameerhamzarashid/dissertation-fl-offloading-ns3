#!/usr/bin/env bash
set -euo pipefail

# Defaults
BUILD_DIR="build"
BUILD_TYPE="Release"
NS3_DIR="${NS3_DIR:-${ns3_DIR:-}}"

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/build_ns3_module.sh                      # BuildType=Release, ns3_DIR from env (NS3_DIR or ns3_DIR)
  ./scripts/build_ns3_module.sh Release              # BuildType=Release, ns3_DIR from env
  ./scripts/build_ns3_module.sh Debug                # BuildType=Debug, ns3_DIR from env
  ./scripts/build_ns3_module.sh Release <ns3_DIR>    # explicit ns3_DIR
  ./scripts/build_ns3_module.sh <ns3_DIR>            # BuildType=Release, explicit ns3_DIR

ns3_DIR must point to a dir that contains both: ns3Config.cmake and ns3Targets.cmake
Examples:
  export NS3_DIR=/path/to/ns-3/build/install/lib/cmake/ns3
  ./scripts/build_ns3_module.sh Release "$NS3_DIR"
USAGE
}

# Parse args
if [[ $# -gt 0 ]]; then
  case "$1" in
    Release|Debug|RelWithDebInfo|MinSizeRel)
      BUILD_TYPE="$1"; shift
      ;;
  esac
fi

if [[ $# -gt 0 ]]; then
  NS3_DIR="$1"; shift
fi

if [[ -z "${NS3_DIR}" ]]; then
  echo "ERROR: ns3_DIR not set. Set NS3_DIR env or pass it as an argument."
  usage
  exit 2
fi

if [[ ! -f "${NS3_DIR}/ns3Config.cmake" || ! -f "${NS3_DIR}/ns3Targets.cmake" ]]; then
  echo "ERROR: ns3_DIR='${NS3_DIR}' does not contain ns3Config.cmake and ns3Targets.cmake"
  exit 3
fi

echo "==> Build type    : ${BUILD_TYPE}"
echo "==> ns3_DIR       : ${NS3_DIR}"
echo "==> Build directory: ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -Dns3_DIR="${NS3_DIR}"

cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -- -j"$(nproc)"
echo "==> Build completed. Binary at ${BUILD_DIR}/run_mec_scenario"
