"""Logical CSV views and per-run materialization."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any, Dict

from .store import Store


def _source_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(metadata.get("source_metadata") or {key: value for key, value in metadata.items() if key not in {"view", "source_metadata"}})


def configure_dataset(store: Store, dataset: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    current = dataset["metadata"]
    source = _source_metadata(current)
    current_view = current.get("view") or {"columns": list(source["files"][0]["columns"]), "renames": {}}
    current_names = {column: current_view.get("renames", {}).get(column) or column for column in current_view["columns"]}
    reverse = {output: original for original, output in current_names.items()}
    selected_outputs = [str(value) for value in config.get("columns", [])]
    requested_renames = {str(key): str(value).strip() for key, value in config.get("renames", {}).items()}
    if not selected_outputs:
        raise ValueError("Keep at least one column")
    id_source = next((column for column, output in current_names.items() if output == dataset["id_column"]), None)
    if id_source is None:
        raise ValueError("The active dataset schema no longer contains its ID column")
    if config.get("schema_mode") == "source":
        source_columns = list(source["files"][0]["columns"])
        if len(selected_outputs) != len(set(selected_outputs)) or any(value not in source_columns for value in selected_outputs):
            raise ValueError("The selected columns do not match the source dataset schema")
        selected_sources = selected_outputs
        renames = {column: requested_renames.get(column) or current_names.get(column) or column for column in selected_sources}
    else:
        if len(selected_outputs) != len(set(selected_outputs)) or any(value not in reverse for value in selected_outputs):
            raise ValueError("The selected columns do not match the active dataset schema")
        selected_sources = [reverse[value] for value in selected_outputs]
        renames = {column: requested_renames.get(current_names[column]) or current_names[column] for column in selected_sources}
    if id_source not in selected_sources:
        raise ValueError(f"The ID column {dataset['id_column']!r} must be included")
    if renames[id_source] != dataset["id_column"]:
        raise ValueError("The ID column cannot be renamed because it links embeddings to targets")
    output_columns = [renames[column] for column in selected_sources]
    if any(not value for value in output_columns) or len(output_columns) != len(set(output_columns)):
        raise ValueError("Every output column name must be non-empty and unique")

    metadata = copy.deepcopy(source)
    for item in metadata["files"]:
        item["columns"] = output_columns
        item["numeric_columns"] = [renames[column] for column in selected_sources if column in item.get("numeric_columns", [])]
        item["empty_sample_columns"] = [renames[column] for column in selected_sources if column in item.get("empty_sample_columns", [])]
        item["examples"] = [{renames[column]: row.get(column, "") for column in selected_sources[:8]} for row in item.get("examples", [])]
        if dataset["kind"] == "embeddings":
            item["feature_columns"] = [column for column in item["numeric_columns"] if column.startswith(dataset["feature_prefix"])]
            item["embedding_dim"] = len(item["feature_columns"])
    if dataset["kind"] == "embeddings":
        metadata["dimensions"] = sorted({item["embedding_dim"] for item in metadata["files"]})
    else:
        metadata["target_columns"] = metadata["files"][0]["numeric_columns"]
    metadata["source_metadata"] = source
    metadata["view"] = {"columns": selected_sources, "renames": renames}
    return store.update_dataset(dataset["id"], str(config.get("name", "")).strip() or dataset["name"], renames[id_source], metadata)


def apply_dataset_view(dataset: Dict[str, Any], records: list[Dict[str, Any]]) -> tuple[list[str], list[Dict[str, Any]]]:
    metadata = dataset["metadata"]
    view = metadata.get("view")
    if not view:
        return list(metadata["files"][0]["columns"]), records
    columns = [view["renames"].get(column) or column for column in view["columns"]]
    return columns, [{view["renames"].get(column) or column: row.get(column, "") for column in view["columns"]} for row in records]


def materialize_dataset_view(dataset: Dict[str, Any], root: Path) -> list[Path]:
    metadata = dataset["metadata"]
    source = _source_metadata(metadata)
    view = metadata.get("view")
    paths = [Path(item["path"]) for item in source["files"]]
    if not view:
        return paths
    output_columns = [view["renames"].get(column) or column for column in view["columns"]]
    destinations = []
    for index, source_path in enumerate(paths):
        model = source["files"][index].get("model_name")
        destination = root / (model or "targets") / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open(newline="", errors="replace") as input_handle, destination.open("w", newline="") as output_handle:
            reader = csv.DictReader(input_handle)
            writer = csv.DictWriter(output_handle, fieldnames=output_columns)
            writer.writeheader()
            for row in reader:
                writer.writerow({view["renames"].get(column) or column: row.get(column, "") for column in view["columns"]})
        destinations.append(destination)
    return destinations
