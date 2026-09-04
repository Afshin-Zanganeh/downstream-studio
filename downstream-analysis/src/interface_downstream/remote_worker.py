"""Small, non-interactive entry point installed on Slurm clusters.

The worker accepts JSON produced by Downstream Studio.  It deliberately exposes
only inspection and probe execution; it is not a general remote shell.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _probe_command(config: dict[str, Any]) -> list[str]:
    command = [sys.executable, "-m", "interface_downstream.cli", "--embeddings"]
    command.extend(map(str, config["embeddings"]))
    command.extend([
        "--targets-csv", str(config["targets_csv"]),
        "--output-dir", str(config["output_dir"]),
        "--id-column", str(config.get("id_column", "id")),
        "--feature-prefix", str(config.get("feature_prefix", "f")),
        "--target-columns", ",".join(config["target_columns"]),
        "--probes", ",".join(config["probes"]),
        "--train-fractions", ",".join(map(str, config["train_fractions"])),
        "--epochs", ",".join(map(str, config["epochs"])),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--weight-decay", str(config["weight_decay"]),
        "--hidden-dim", str(config["hidden_dim"]),
        "--linear-width", str(config.get("linear_width", 0)),
        "--hidden-layers", ",".join(str(layer["units"]) for layer in config.get("layers", [])),
        "--activations", ",".join(layer["activation"] for layer in config.get("layers", [])) or "gelu",
        "--dropout", str(config["dropout"]),
        "--seed", str(config["seed"]),
        "--device", str(config["device"]),
        "--log-every", str(config.get("log_every", 10)),
    ])
    if config.get("save_predictions"):
        command.append("--save-predictions")
    return command


def run(config_path: str) -> int:
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = _probe_command(config)
    (output / "studio_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    return subprocess.run(command, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Downstream Studio HPC worker")
    parser.add_argument("config", help="Validated JSON job configuration")
    args = parser.parse_args()
    raise SystemExit(run(args.config))


if __name__ == "__main__":
    main()
