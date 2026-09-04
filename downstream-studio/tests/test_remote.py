import tarfile
from pathlib import Path

from downstream_studio.remote import build_worker_bundle
from downstream_studio.store import Store


def test_worker_bundle_contains_installable_analysis_package():
    archive, version = build_worker_bundle()
    try:
        assert len(version) == 12
        with tarfile.open(archive, "r:gz") as bundle:
            names = set(bundle.getnames())
        assert "downstream-analysis/pyproject.toml" in names
        assert "downstream-analysis/src/interface_downstream/remote_worker.py" in names
    finally:
        archive.unlink(missing_ok=True)


def test_remote_dataset_keeps_hpc_path_and_schema_migrates(tmp_path):
    store = Store(tmp_path / "studio")
    project = store.create_project("Remote")
    remote_path = "/data/horse/ws/project/embeddings"
    dataset = store.create_dataset(
        project["id"], "HPC embeddings", "embeddings", "path", remote_path,
        "id", "f", {"location": "hpc", "files": [], "models": [], "dimensions": []},
    )
    assert dataset["path"] == remote_path
    with store.connect() as db:
        experiment_columns = {row[1] for row in db.execute("PRAGMA table_info(experiments)")}
        connection_columns = {row[1] for row in db.execute("PRAGMA table_info(hpc_connections)")}
    assert {"remote_job_id", "remote_output_dir"} <= experiment_columns
    assert {"auth_method", "remote_python", "worker_version"} <= connection_columns
