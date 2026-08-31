"""Benchmark results and serialization."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class BenchmarkResult:
    metrics: List[Dict[str, object]]
    history: List[Dict[str, object]]
    predictions: List[Dict[str, object]]
    metadata: Dict[str, object]

    def best_per_target(self) -> List[Dict[str, object]]:
        best: Dict[str, Dict[str, object]] = {}
        for row in self.metrics:
            target = str(row["target"])
            score = float(row["r2"])
            if target not in best or score > float(best[target]["r2"]):
                best[target] = row
        return [best[name] for name in sorted(best)]

    def write(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        _write_rows(output / "probe_results.csv", self.metrics)
        _write_rows(output / "training_history.csv", self.history)
        _write_rows(output / "best_probe_per_target.csv", self.best_per_target())
        _write_rows(output / "test_predictions.csv", self.predictions)
        with (output / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2)

