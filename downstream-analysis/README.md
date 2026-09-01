# Interface downstream analysis

An independent package for testing which numerical interface features can be
re-predicted from learned embeddings. It does not import or depend on
Graphormer, UniMol, PLIP, PostgreSQL, or a particular file format.

The core API consumes in-memory objects:

```python
import numpy as np
from interface_downstream import BenchmarkConfig, EmbeddingSet, TargetSet, run_benchmark

embedding_sets = {
    "pdb_model_1": EmbeddingSet(ids=ids, values=pdb_embeddings_1),
    "pdb_model_2": EmbeddingSet(ids=ids, values=pdb_embeddings_2),
    "af_model": EmbeddingSet(ids=ids, values=af_embeddings),
}
targets = TargetSet(
    ids=ids,
    values={
        "number_nodes": number_nodes,
        "number_hbonds": number_hbonds,
        "mean_distance": mean_distance,
    },
)

result = run_benchmark(
    embedding_sets,
    targets,
    BenchmarkConfig(
        probes=("linear", "mlp"),
        train_fractions=(0.5, 0.8),
        evaluation_epochs=(250, 500, 1500),
    ),
)

result.write("results")
```

`ids` may be a list, NumPy array, database primary keys converted to strings,
or any other sequence of unique values. Load the data however you prefer, then
construct `EmbeddingSet` and `TargetSet`.

## Installation

```bash
cd downstream-analysis
python -m pip install -e .
```

Install the optional CSV adapter:

```bash
python -m pip install -e '.[csv]'
```

## CSV command

Embedding files must contain an ID column and numeric columns such as `f1` to
`f264`. The target file contains the same ID and one or more numeric interface
features.

```bash
interface-probes \
  --embeddings '/path/to/af_random_100k_mixed/*/embeddings*.csv' \
  --targets-csv interface_features.csv \
  --output-dir results \
  --id-column id \
  --probes linear,mlp \
  --train-fractions 0.5,0.8 \
  --epochs 250,500,1500 \
  --device cuda
```

To reproduce a 50/50, 1500-epoch linear-probe experiment:

```bash
interface-probes \
  --embeddings '/path/to/af_random_100k_mixed/*/embeddings*.csv' \
  --targets-csv interface_features.csv \
  --output-dir results \
  --probes linear \
  --train-fractions 0.5 \
  --epochs 1500
```

## Outputs

- `probe_results.csv`: held-out metrics for every experiment.
- `best_probe_per_target.csv`: highest-R² configuration per target.
- `training_history.csv`: normalized training loss.
- `test_predictions.csv`: optional held-out predictions.
- `run_metadata.json`: settings and input summary.

All embedding models are compared on the intersection of their IDs. For a
given target and train fraction, every model receives exactly the same train
and test IDs. Normalization statistics are fitted only on the training split.

When embedding CSVs have identical filenames inside separate run folders, the
run-folder name is used as `embedding_model` in the result files.

The `examples/` directory contains a short local smoke test using the manifest's
`nodes` column and an HPC template for a complete database-feature export.

## Export interface features from PostgreSQL

Install the PostgreSQL adapter:

```bash
python -m pip install -e '.[postgres]'
```

Export every column from database rows whose IDs appear in an embedding CSV:

```bash
interface-export-postgres \
  --dsn "$PDRUG_DSN" \
  --input-csv /path/to/embeddings.csv \
  --table interface_af \
  --output-csv interface_features.csv
```

The command uses `id` in both files by default. Override these with
`--input-id-column` and `--db-id-column`. The output preserves embedding-CSV
ID order, contains the database table's complete header, and is written through
a temporary `.partial` file. Missing IDs are reported separately as
`interface_features.csv.missing_ids.csv`.
