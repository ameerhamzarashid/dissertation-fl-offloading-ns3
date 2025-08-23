#!/usr/bin/env bash
set -euo pipefail

# Sync NS-3 CSVs into project-root data/raw_logs for Python experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/ns3_module/data/raw_logs"
DST_DIR="${ROOT_DIR}/data/raw_logs"

mkdir -p "${DST_DIR}"

if compgen -G "${SRC_DIR}/*.csv" > /dev/null; then
  cp -f "${SRC_DIR}"/*.csv "${DST_DIR}/"
  echo "Synced CSVs to ${DST_DIR}"
else
  echo "No CSVs found in ${SRC_DIR} (run NS-3 first)."
fi
