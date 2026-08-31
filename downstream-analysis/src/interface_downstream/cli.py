"""CSV command-line interface for the data-agnostic package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.csv import load_embedding_csvs, load_target_csv
from .config import BenchmarkConfig, ProbeConfig
from .runner import run_benchmark


def _items(value: str):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe interface features from one or more embedding CSVs.")
    parser.add_argument("--embeddings", nargs="+", required=True)
    parser.add_argument("--targets-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--feature-prefix", default="f")
    parser.add_argument("--target-columns", default=None)
    parser.add_argument("--exclude-target-columns", default="")
    parser.add_argument("--probes", default="linear")
    parser.add_argument("--train-fractions", default="0.5")
    parser.add_argument("--epochs", default="1500")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--linear-width", type=int, default=0)
    parser.add_argument("--hidden-layers", default="")
    parser.add_argument("--activations", default="gelu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    embeddings, paths = load_embedding_csvs(args.embeddings, args.id_column, args.feature_prefix)
    targets = load_target_csv(
        args.targets_csv,
        id_column=args.id_column,
        target_columns=_items(args.target_columns) if args.target_columns else None,
        exclude_columns=_items(args.exclude_target_columns),
    )
    config = BenchmarkConfig(
        probes=_items(args.probes),
        train_fractions=tuple(float(value) for value in _items(args.train_fractions)),
        evaluation_epochs=tuple(int(value) for value in _items(args.epochs)),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        log_every=args.log_every,
        save_predictions=args.save_predictions,
        probe=ProbeConfig(
            hidden_dim=args.hidden_dim,
            linear_width=args.linear_width,
            hidden_layers=tuple(int(value) for value in _items(args.hidden_layers)),
            activations=_items(args.activations),
            dropout=args.dropout,
        ),
    )
    def progress(event):
        print("STUDIO_PROGRESS " + json.dumps(event), flush=True)

    result = run_benchmark(embeddings, targets, config, progress_callback=progress)
    result.metadata.update({
        "embedding_files": [str(path) for path in paths],
        "targets_csv": str(Path(args.targets_csv).resolve()),
    })
    result.write(args.output_dir)
    print(f"Completed {len(result.metrics)} evaluations.")
    print(f"Results written to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
