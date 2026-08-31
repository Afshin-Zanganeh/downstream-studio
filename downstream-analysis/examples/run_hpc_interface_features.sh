#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /path/to/interface_features.csv /path/to/output_directory" >&2
  exit 2
fi

TARGETS_CSV="$1"
OUTPUT_DIR="$2"
EMBEDDING_ROOT="/data/horse/ws/lubu348b-plip_graph/embeddings/runs/af_random_100k_mixed"

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

