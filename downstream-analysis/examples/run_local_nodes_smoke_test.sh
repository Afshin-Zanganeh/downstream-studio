#!/usr/bin/env bash
set -euo pipefail

# The bundled record manifest only contains one usable numeric target: nodes.
# This is a pipeline smoke test, not the complete supervisor-requested analysis.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

interface-probes \
  --embeddings "${PROJECT_ROOT}/af_random_100k_mixed/*/embeddings*.csv" \
  --targets-csv "${PROJECT_ROOT}/af_random_100k_mixed/af_100k_40e/records_af_random_100k_mixed.csv.gz" \
  --output-dir "${PROJECT_ROOT}/downstream-analysis/results/nodes_smoke_test" \
  --id-column id \
  --target-columns nodes \
  --probes linear \
  --train-fractions 0.5 \
  --epochs 50 \
  --device auto

