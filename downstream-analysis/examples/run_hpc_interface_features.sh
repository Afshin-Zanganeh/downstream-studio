#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 /path/to/embeddings /path/to/interface_features.csv /path/to/output_directory" >&2
  exit 2
fi

EMBEDDING_ROOT="$1"
TARGETS_CSV="$2"
OUTPUT_DIR="$3"

interface-probes \
  --embeddings "${EMBEDDING_ROOT}/*/embeddings*.csv" \
  --targets-csv "${TARGETS_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --id-column id \
  --probes linear \
  --train-fractions 0.5,0.8 \
  --epochs 250,500,1500 \
  --batch-size 1024 \
  --device cuda
