"""Fast, bounded CSV inspection for embeddings and target tables."""

from __future__ import annotations

import csv
import glob
import math
from pathlib import Path
from typing import Any, Dict, Iterable


def resolve_embedding_paths(value: str) -> list[Path]:
    path = Path(value).expanduser()
    if path.is_dir():
        matches = sorted(path.glob("**/embeddings*.csv"))
    else:
        matches = [Path(item) for item in sorted(glob.glob(str(path)))]
    return [item.resolve() for item in matches if item.is_file()]


def _is_number(value: str) -> bool:
    if value.strip() == "":
        return True
    try:
        return math.isfinite(float(value)) or value.lower() in {"nan", "inf", "-inf"}
    except ValueError:
        return False


def inspect_one_csv(path: Path, id_column: str = "id", sample_rows: int = 500) -> Dict[str, Any]:
    size = path.stat().st_size
    with path.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if id_column not in columns:
            raise ValueError(f"Missing ID column {id_column!r} in {path}")
        numeric = {column: True for column in columns if column != id_column}
        nonempty = {column: 0 for column in columns if column != id_column}
        examples = []
        count = 0
        for row in reader:
            count += 1
            if len(examples) < 3:
                examples.append({key: row.get(key, "") for key in columns[:8]})
            if count <= sample_rows:
                for column in numeric:
                    value = row.get(column, "")
                    if value.strip():
                        nonempty[column] += 1
                    if not _is_number(value):
                        numeric[column] = False
        sampled = min(count, sample_rows)
    numeric_columns = [column for column in columns if numeric.get(column) and nonempty.get(column, 0)]
    empty_columns = [column for column in columns if column != id_column and nonempty.get(column, 0) == 0]
    return {
        "path": str(path), "filename": path.name, "size_bytes": size,
        "columns": columns, "numeric_columns": numeric_columns,
        "empty_sample_columns": empty_columns, "sampled_rows": sampled,
        "at_least_rows": count, "examples": examples,
    }


def inspect_dataset(kind: str, value: str, id_column: str = "id", feature_prefix: str = "f") -> Dict[str, Any]:
    paths = resolve_embedding_paths(value) if kind == "embeddings" else [Path(value).expanduser().resolve()]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"No readable CSV files found for {value!r}")
    files = [inspect_one_csv(path, id_column) for path in paths]
    result: Dict[str, Any] = {"kind": kind, "files": files, "file_count": len(files)}
    if kind == "embeddings":
        for item in files:
            item["feature_columns"] = [c for c in item["numeric_columns"] if c.startswith(feature_prefix)]
            item["embedding_dim"] = len(item["feature_columns"])
            item["model_name"] = Path(item["path"]).parent.name
        result["models"] = [item["model_name"] for item in files]
        result["dimensions"] = sorted({item["embedding_dim"] for item in files})
    else:
        result["target_columns"] = files[0]["numeric_columns"]
    return result
