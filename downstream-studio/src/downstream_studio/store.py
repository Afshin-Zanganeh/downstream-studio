"""SQLite persistence and workspace layout."""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.uploads = self.root / "uploads"
        self.runs = self.root / "runs"
        self.trash = self.root / "trash"
        self.uploads.mkdir(exist_ok=True)
        self.runs.mkdir(exist_ok=True)
        self.trash.mkdir(exist_ok=True)
        self.database = self.root / "studio.sqlite3"
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('embeddings', 'targets')),
                    source_type TEXT NOT NULL CHECK(source_type IN ('path', 'upload')),
                    path TEXT NOT NULL,
                    id_column TEXT NOT NULL DEFAULT 'id',
                    feature_prefix TEXT NOT NULL DEFAULT 'f',
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    pid INTEGER,
                    return_code INTEGER,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS hpc_connections (
                    project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL DEFAULT 'TU Dresden Capella',
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL DEFAULT 22,
                    username TEXT NOT NULL,
                    remote_workspace TEXT NOT NULL DEFAULT '',
                    last_status TEXT NOT NULL DEFAULT 'not_tested',
                    last_message TEXT NOT NULL DEFAULT '',
                    last_tested_at TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        item = dict(row)
        for key in ("metadata_json", "config_json"):
            if key in item:
                item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        project_id = uuid.uuid4().hex[:12]
        with self.connect() as db:
            db.execute(
                "INSERT INTO projects VALUES (?, ?, ?, ?)",
                (project_id, name.strip(), description.strip(), now()),
            )
        return self.get_project(project_id)  # type: ignore[return-value]

    def list_projects(self) -> list[Dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT p.*,
                    (SELECT COUNT(*) FROM datasets d WHERE d.project_id=p.id) dataset_count,
                    (SELECT COUNT(*) FROM experiments e WHERE e.project_id=p.id) experiment_count
                    FROM projects p ORDER BY created_at DESC"""
            ).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[misc]

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        return self._row(row)

    def delete_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        project = self.get_project(project_id)
        if not project:
            return None
        experiments = self.list_experiments(project_id)
        if any(item["status"] in {"queued", "running"} for item in experiments):
            raise ValueError("Cancel queued or running experiments before deleting this project")
        with self.connect() as db:
            db.execute("DELETE FROM projects WHERE id=?", (project_id,))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = self.trash / f"project-{project_id}-{timestamp}"
        moved = False
        upload_root = self.uploads / project_id
        if upload_root.exists():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.move(str(upload_root), str(destination / "uploads"))
            moved = True
        for experiment in experiments:
            output = Path(experiment["output_dir"])
            if output.is_dir():
                run_destination = destination / "runs"
                run_destination.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output), str(run_destination / output.name))
                moved = True
        if moved:
            project["recoverable_from"] = str(destination)
        return project

    def create_dataset(
        self, project_id: str, name: str, kind: str, source_type: str, path: Path,
        id_column: str, feature_prefix: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        dataset_id = uuid.uuid4().hex[:12]
        with self.connect() as db:
            db.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (dataset_id, project_id, name.strip(), kind, source_type, str(path.resolve()),
                 id_column, feature_prefix, json.dumps(metadata), now()),
            )
        return self.get_dataset(dataset_id)  # type: ignore[return-value]

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return self._row(row)

    def list_datasets(self, project_id: str) -> list[Dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM datasets WHERE project_id=? ORDER BY created_at", (project_id,)
            ).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[misc]

    def update_dataset(self, dataset_id: str, name: str, id_column: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        with self.connect() as db:
            db.execute(
                "UPDATE datasets SET name=?, id_column=?, metadata_json=? WHERE id=?",
                (name.strip(), id_column, json.dumps(metadata), dataset_id),
            )
        return self.get_dataset(dataset_id)  # type: ignore[return-value]

    def delete_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        with self.connect() as db:
            running = db.execute(
                "SELECT id FROM experiments WHERE status IN ('queued','running') AND (json_extract(config_json,'$.embeddings_dataset_id')=? OR json_extract(config_json,'$.targets_dataset_id')=?) LIMIT 1",
                (dataset_id, dataset_id),
            ).fetchone()
            if running:
                raise ValueError("This dataset is used by a queued or running experiment")
            db.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
        path = Path(dataset["path"])
        try:
            path.resolve().relative_to(self.uploads.resolve())
            if path.exists():
                destination = self.trash / f"dataset-{dataset_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                shutil.move(str(path), str(destination))
                dataset["recoverable_from"] = str(destination)
        except ValueError:
            pass
        return dataset

    def get_hpc_connection(self, project_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM hpc_connections WHERE project_id=?", (project_id,)).fetchone()
        return self._row(row)

    def save_hpc_connection(self, project_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = now()
        with self.connect() as db:
            db.execute(
                """INSERT INTO hpc_connections
                   (project_id,name,host,port,username,remote_workspace,last_status,last_message,last_tested_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(project_id) DO UPDATE SET
                     name=excluded.name, host=excluded.host, port=excluded.port,
                     username=excluded.username, remote_workspace=excluded.remote_workspace,
                     last_status=excluded.last_status, last_message=excluded.last_message,
                     last_tested_at=excluded.last_tested_at, updated_at=excluded.updated_at""",
                (project_id, values.get("name", "TU Dresden Capella"), values["host"], int(values.get("port", 22)),
                 values["username"], values.get("remote_workspace", ""), values.get("last_status", "not_tested"),
                 values.get("last_message", ""), values.get("last_tested_at"), timestamp),
            )
        return self.get_hpc_connection(project_id)  # type: ignore[return-value]

    def create_experiment(self, project_id: str, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        experiment_id = uuid.uuid4().hex[:12]
        output_dir = self.runs / experiment_id
        output_dir.mkdir(parents=True)
        log_path = output_dir / "run.log"
        with self.connect() as db:
            db.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, NULL, NULL)",
                (experiment_id, project_id, name.strip(), "queued", json.dumps(config),
                 str(output_dir), str(log_path), now()),
            )
        return self.get_experiment(experiment_id)  # type: ignore[return-value]

    def update_experiment(self, experiment_id: str, **values: Any) -> None:
        allowed = {"status", "pid", "return_code", "error", "started_at", "finished_at"}
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.connect() as db:
            db.execute(
                f"UPDATE experiments SET {assignments} WHERE id=?", (*values.values(), experiment_id)
            )

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        return self._row(row)

    def list_experiments(self, project_id: str) -> list[Dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM experiments WHERE project_id=? ORDER BY created_at DESC", (project_id,)
            ).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[misc]

    def delete_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
        output = Path(experiment["output_dir"])
        if output.is_dir():
            destination = self.trash / f"{experiment_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            shutil.move(str(output), str(destination))
            experiment["recoverable_from"] = str(destination)
        with self.connect() as db:
            db.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
        return experiment
