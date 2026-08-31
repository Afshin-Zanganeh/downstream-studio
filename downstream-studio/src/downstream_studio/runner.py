"""Background execution adapter for interface_downstream."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .store import Store, now
from .transform_csv import materialize_dataset_view


class RunManager:
    def __init__(self, store: Store):
        self.store = store
        self.processes: Dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()

    def command(self, experiment: Dict[str, Any]) -> list[str]:
        config = experiment["config"]
        embeddings = self.store.get_dataset(config["embeddings_dataset_id"])
        targets = self.store.get_dataset(config["targets_dataset_id"])
        if not embeddings or not targets:
            raise ValueError("An input dataset no longer exists")
        if embeddings["id_column"] != targets["id_column"]:
            raise ValueError("Embedding and target datasets must use the same ID column")
        view_root = Path(experiment["output_dir"]) / "studio_inputs"
        embedding_paths = materialize_dataset_view(embeddings, view_root / "embeddings")
        target_paths = materialize_dataset_view(targets, view_root / "targets")
        cmd = [sys.executable, "-m", "interface_downstream.cli", "--embeddings"]
        selected_models = set(config.get("embedding_models") or [])
        embedding_files = [{**item, "path": str(embedding_paths[index])} for index, item in enumerate(embeddings["metadata"]["files"])]
        if selected_models:
            embedding_files = [file for file in embedding_files if file.get("model_name") in selected_models]
        if not embedding_files:
            raise ValueError("No embedding files match the selected models")
        cmd.extend(file["path"] for file in embedding_files)
        cmd.extend([
            "--targets-csv", str(target_paths[0]), "--output-dir", experiment["output_dir"],
            "--id-column", embeddings["id_column"],
            "--feature-prefix", config.get("feature_prefix", "f"),
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
            "--seed", str(config["seed"]), "--device", config["device"],
            "--log-every", str(config.get("log_every", 10)),
        ])
        if config.get("save_predictions"):
            cmd.append("--save-predictions")
        return cmd

    def start(self, experiment_id: str) -> None:
        experiment = self.store.get_experiment(experiment_id)
        if not experiment or experiment["status"] != "queued":
            raise ValueError("Experiment is missing or is not queued")
        thread = threading.Thread(target=self._execute, args=(experiment_id,), daemon=True)
        thread.start()

    def _execute(self, experiment_id: str) -> None:
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            return
        try:
            cmd = self.command(experiment)
            output = Path(experiment["output_dir"])
            (output / "studio_config.json").write_text(json.dumps(experiment["config"], indent=2))
            (output / "command.json").write_text(json.dumps(cmd, indent=2))
            env = os.environ.copy()
            sibling_src = Path(__file__).resolve().parents[3] / "downstream-analysis" / "src"
            if sibling_src.exists():
                env["PYTHONPATH"] = str(sibling_src) + os.pathsep + env.get("PYTHONPATH", "")
            with Path(experiment["log_path"]).open("w", buffering=1) as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                with self.lock:
                    self.processes[experiment_id] = process
                self.store.update_experiment(experiment_id, status="running", pid=process.pid, started_at=now())
                code = process.wait()
            status = "completed" if code == 0 else "failed"
            self.store.update_experiment(
                experiment_id, status=status, return_code=code, finished_at=now(),
                error=None if code == 0 else self.tail_log(experiment_id, 30),
            )
        except Exception as error:
            self.store.update_experiment(experiment_id, status="failed", error=str(error), finished_at=now())
        finally:
            with self.lock:
                self.processes.pop(experiment_id, None)

    def cancel(self, experiment_id: str) -> bool:
        with self.lock:
            process = self.processes.get(experiment_id)
        if not process or process.poll() is not None:
            return False
        os.killpg(process.pid, signal.SIGTERM)
        self.store.update_experiment(experiment_id, status="cancelled", finished_at=now())
        return True

    def tail_log(self, experiment_id: str, lines: int = 100) -> str:
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            return ""
        path = Path(experiment["log_path"])
        if not path.exists():
            return ""
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])

    def progress(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        config = experiment["config"]
        expected = (
            max(1, len(config.get("embedding_models") or []))
            * len(config.get("target_columns", []))
            * len(config.get("probes", []))
            * len(config.get("train_fractions", []))
            * len(config.get("epochs", []))
        )
        path = Path(experiment["log_path"])
        events = []
        if path.exists():
            for line in path.read_text(errors="replace").splitlines():
                if line.startswith("STUDIO_PROGRESS "):
                    try:
                        events.append(json.loads(line.removeprefix("STUDIO_PROGRESS ")))
                    except json.JSONDecodeError:
                        pass
        completed = len(events)
        elapsed = None
        remaining = None
        if experiment.get("started_at"):
            from datetime import datetime, timezone
            started = datetime.fromisoformat(experiment["started_at"])
            finished = datetime.fromisoformat(experiment["finished_at"]) if experiment.get("finished_at") else datetime.now(timezone.utc)
            elapsed = max(0.0, (finished - started).total_seconds())
            if completed and completed < expected:
                remaining = elapsed / completed * (expected - completed)
        return {
            "completed": completed, "expected": expected,
            "percent": round(100 * completed / expected, 1) if expected else 0,
            "elapsed_seconds": elapsed, "estimated_remaining_seconds": remaining,
            "latest": events[-1] if events else None,
        }


def read_csv_records(path: Path, limit: Optional[int] = None) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            converted: Dict[str, Any] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (ValueError, TypeError):
                    converted[key] = value
            output.append(converted)
            if limit and len(output) >= limit:
                break
    return output
