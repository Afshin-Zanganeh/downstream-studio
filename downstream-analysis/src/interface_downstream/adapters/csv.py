"""Pandas-based CSV adapter kept separate from the benchmark core."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..data import EmbeddingSet, TargetSet


def _unique_ids(frame: pd.DataFrame, column: str, source: str) -> None:
    if column not in frame:
        raise ValueError(f"{source} does not contain ID column {column!r}.")
    if frame[column].isna().any() or frame[column].astype(str).duplicated().any():
        raise ValueError(f"{source} IDs must be present and unique.")


def expand_paths(patterns: Sequence[str]) -> List[Path]:
    paths = sorted({Path(match).resolve() for pattern in patterns for match in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("No embedding CSVs matched the supplied paths/patterns.")
    return paths


def load_embedding_csv(path: str | Path, id_column: str = "id", feature_prefix: str = "f") -> EmbeddingSet:
    path = Path(path)
    frame = pd.read_csv(path)
    _unique_ids(frame, id_column, str(path))
    columns = [
        column for column in frame.columns
        if column != id_column and column.startswith(feature_prefix) and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise ValueError(f"No numeric {feature_prefix!r}-prefixed columns in {path}.")
    columns.sort(key=lambda column: (0, int(column[len(feature_prefix):])) if column[len(feature_prefix):].isdigit() else (1, column))
    return EmbeddingSet(
        ids=frame[id_column].astype(str).to_numpy(),
        values=frame[columns].to_numpy(dtype="float32"),
        feature_names=columns,
    )


def load_embedding_csvs(
    patterns: Sequence[str], id_column: str = "id", feature_prefix: str = "f"
) -> Tuple[Dict[str, EmbeddingSet], List[Path]]:
    paths = expand_paths(patterns)
    datasets: Dict[str, EmbeddingSet] = {}
    for path in paths:
        # Embedding exports commonly use the same CSV filename in separate run
        # directories. The parent folder is therefore the informative model/run
        # name (e.g. af_100k_40e, pdb_100k_37e, unimol_half).
        name = path.parent.name or path.stem
        counter = 2
        while name in datasets:
            name = f"{path.parent.name or path.stem}_{counter}"
            counter += 1
        datasets[name] = load_embedding_csv(path, id_column, feature_prefix)
    return datasets, paths


def load_target_csv(
    path: str | Path,
    id_column: str = "id",
    target_columns: Optional[Sequence[str]] = None,
    exclude_columns: Sequence[str] = (),
) -> TargetSet:
    path = Path(path)
    frame = pd.read_csv(path)
    _unique_ids(frame, id_column, str(path))
    if target_columns is None:
        excluded = set(exclude_columns) | {id_column}
        target_columns = [
            column for column in frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        ]
    missing = sorted(set(target_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Target columns missing from {path}: {missing}")
    return TargetSet(
        ids=frame[id_column].astype(str).to_numpy(),
        values={column: pd.to_numeric(frame[column], errors="coerce").to_numpy() for column in target_columns},
    )
