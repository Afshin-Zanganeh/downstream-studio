"""Data-agnostic in-memory inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def _normalize_ids(ids: Sequence[object], name: str) -> np.ndarray:
    values = np.asarray([str(value) for value in ids], dtype=object)
    if values.ndim != 1:
        raise ValueError(f"{name} IDs must be one-dimensional.")
    if len(set(values.tolist())) != len(values):
        raise ValueError(f"{name} IDs must be unique.")
    return values


@dataclass(frozen=True)
class EmbeddingSet:
    """One model's embeddings, independent of their original storage."""

    ids: Sequence[object]
    values: np.ndarray
    feature_names: Optional[Sequence[str]] = None

    def validated(self, name: str = "embedding") -> "EmbeddingSet":
        ids = _normalize_ids(self.ids, name)
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"{name} values must have shape [samples, dimensions].")
        if values.shape[0] != len(ids):
            raise ValueError(f"{name} has {len(ids)} IDs but {values.shape[0]} rows.")
        if values.shape[1] < 1 or not np.isfinite(values).all():
            raise ValueError(f"{name} embeddings must be finite and non-empty.")
        names = tuple(self.feature_names) if self.feature_names is not None else None
        if names is not None and len(names) != values.shape[1]:
            raise ValueError(f"{name} feature_names length does not match embedding dimensions.")
        return EmbeddingSet(ids=ids, values=values, feature_names=names)


@dataclass(frozen=True)
class TargetSet:
    """Multiple scalar regression targets associated with interface IDs."""

    ids: Sequence[object]
    values: Mapping[str, Sequence[float]]

    def validated(self) -> "TargetSet":
        ids = _normalize_ids(self.ids, "target")
        if not self.values:
            raise ValueError("At least one target feature is required.")
        targets: Dict[str, np.ndarray] = {}
        for name, values in self.values.items():
            array = np.asarray(values, dtype=np.float32)
            if array.ndim != 1 or len(array) != len(ids):
                raise ValueError(f"Target {name!r} must contain one value per ID.")
            targets[str(name)] = array
        return TargetSet(ids=ids, values=targets)


@dataclass(frozen=True)
class AlignedInputs:
    ids: np.ndarray
    embeddings: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    feature_names: Mapping[str, Optional[Tuple[str, ...]]]


def align_inputs(
    embedding_sets: Mapping[str, EmbeddingSet],
    target_set: TargetSet,
) -> AlignedInputs:
    if not embedding_sets:
        raise ValueError("At least one embedding set is required.")
    validated_embeddings = {
        str(name): dataset.validated(str(name)) for name, dataset in embedding_sets.items()
    }
    validated_targets = target_set.validated()
    common = set(validated_targets.ids.tolist())
    for dataset in validated_embeddings.values():
        common &= set(dataset.ids.tolist())
    if not common:
        raise ValueError("Embedding sets and targets have no shared IDs.")
    ids = np.asarray(sorted(common), dtype=object)

    embeddings: Dict[str, np.ndarray] = {}
    feature_names: Dict[str, Optional[Tuple[str, ...]]] = {}
    for name, dataset in validated_embeddings.items():
        lookup = {value: index for index, value in enumerate(dataset.ids)}
        embeddings[name] = dataset.values[[lookup[value] for value in ids]]
        feature_names[name] = tuple(dataset.feature_names) if dataset.feature_names is not None else None
    target_lookup = {value: index for index, value in enumerate(validated_targets.ids)}
    targets = {
        name: np.asarray(values)[[target_lookup[value] for value in ids]]
        for name, values in validated_targets.values.items()
    }
    return AlignedInputs(ids=ids, embeddings=embeddings, targets=targets, feature_names=feature_names)

