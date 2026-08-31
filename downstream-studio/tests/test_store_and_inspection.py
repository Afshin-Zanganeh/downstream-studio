import csv
from pathlib import Path

import pytest

from downstream_studio.inspect_csv import inspect_dataset
from downstream_studio.server import StudioApp
from downstream_studio.store import Store
from downstream_studio.transform_csv import apply_dataset_view, configure_dataset, materialize_dataset_view


def write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def test_inspection_and_store(tmp_path):
    model = tmp_path / "model_a"
    model.mkdir()
    embeddings = model / "embeddings_test.csv"
    targets = tmp_path / "targets.csv"
    write_csv(embeddings, [["id", "f1", "f2"], ["a", 1, 2], ["b", 3, 4]])
    write_csv(targets, [["id", "score", "text"], ["a", 2.5, "x"], ["b", 3.5, "y"]])

    emb_meta = inspect_dataset("embeddings", str(tmp_path))
    target_meta = inspect_dataset("targets", str(targets))
    assert emb_meta["models"] == ["model_a"]
    assert emb_meta["dimensions"] == [2]
    assert target_meta["target_columns"] == ["score"]

    store = Store(tmp_path / "studio")
    project = store.create_project("Test")
    dataset = store.create_dataset(
        project["id"], "Targets", "targets", "path", targets, "id", "f", target_meta
    )
    experiment = store.create_experiment(project["id"], "Run", {"target_columns": ["score"]})
    assert dataset["metadata"]["target_columns"] == ["score"]
    assert experiment["status"] == "queued"

    app = StudioApp(tmp_path / "browser", [tmp_path])
    listing = app.browse(str(tmp_path))
    assert any(item["name"] == "targets.csv" and item["type"] == "csv" for item in listing["entries"])

    edited = configure_dataset(store, dataset, {
        "name": "Selected targets", "columns": ["id", "score"], "renames": {"score": "binding_score"},
    })
    assert edited["name"] == "Selected targets"
    assert edited["id"] == dataset["id"]
    assert edited["metadata"]["target_columns"] == ["binding_score"]
    with targets.open(newline="") as handle:
        assert next(csv.reader(handle)) == ["id", "score", "text"]
    columns, preview = apply_dataset_view(edited, [{"id": "a", "score": 2.5, "text": "x"}])
    assert columns == ["id", "binding_score"]
    assert preview == [{"id": "a", "binding_score": 2.5}]
    assert edited["metadata"]["source_metadata"]["files"][0]["columns"] == ["id", "score", "text"]

    edited = configure_dataset(store, edited, {
        "name": "Selected targets", "columns": ["id", "score", "text"],
        "renames": {"score": "binding_score"}, "schema_mode": "source",
    })
    columns, preview = apply_dataset_view(edited, [{"id": "a", "score": 2.5, "text": "x"}])
    assert columns == ["id", "binding_score", "text"]
    assert preview == [{"id": "a", "binding_score": 2.5, "text": "x"}]
    materialized = materialize_dataset_view(edited, tmp_path / "run-inputs")
    with materialized[0].open(newline="") as handle:
        assert next(csv.reader(handle)) == ["id", "binding_score", "text"]

    deleted = store.delete_dataset(edited["id"])
    assert deleted["id"] == edited["id"]
    assert store.get_dataset(edited["id"]) is None
    assert targets.exists()  # Path-backed source files are never deleted.

    with pytest.raises(ValueError, match="Cancel queued or running"):
        store.delete_project(project["id"])
    store.update_experiment(experiment["id"], status="completed")
    deleted_project = store.delete_project(project["id"])
    assert deleted_project["id"] == project["id"]
    assert store.get_project(project["id"]) is None
    assert deleted_project.get("recoverable_from")
