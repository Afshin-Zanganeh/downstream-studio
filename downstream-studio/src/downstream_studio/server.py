"""Dependency-light HTTP API and static web server."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from .hpc import (
    HPC_HOSTS, connect_client, forget_password as forget_saved_password,
    load_password as load_saved_password, save_password as store_saved_password, test_connection,
)
from .inspect_csv import inspect_dataset
from .runner import RunManager, read_csv_records
from .store import Store, now
from .transform_csv import apply_dataset_view, configure_dataset
from .remote import browse_remote, deploy_worker, inspect_remote_dataset, preview_remote_csv, worker_status


class StudioApp:
    def __init__(self, workspace: Path, browse_roots: Optional[list[Path]] = None):
        self.store = Store(workspace)
        self.hpc_passwords: Dict[str, str] = {}
        self.runs = RunManager(self.store, self.hpc_client)
        self.web = Path(__file__).with_name("web")
        configured = browse_roots or [Path.cwd()]
        self.browse_roots = sorted({path.expanduser().resolve() for path in configured if path.expanduser().exists()})

    def hpc_client(self, project_id: str):
        profile = self.store.get_hpc_connection(project_id)
        if not profile:
            raise ValueError("Configure the HPC connection for this project first")
        auth_method = profile.get("auth_method", "agent")
        password = ""
        if auth_method == "password":
            password = self.hpc_passwords.get(project_id) or load_saved_password(
                project_id, profile["host"], profile["username"]
            ) or ""
        return connect_client(profile["host"], int(profile["port"]), profile["username"], password, auth_method)

    def browse(self, requested: Optional[str]) -> Dict[str, Any]:
        if not self.browse_roots:
            raise ValueError("No filesystem browsing roots are configured")
        if not requested:
            return {
                "roots": [{"name": root.name or str(root), "path": str(root)} for root in self.browse_roots],
                "path": None, "parent": None, "entries": [],
            }
        current = Path(requested).expanduser().resolve()
        root = next((candidate for candidate in self.browse_roots if current == candidate or candidate in current.parents), None)
        if root is None:
            raise PermissionError("That path is outside the configured browsing roots")
        if not current.is_dir():
            raise ValueError("The requested path is not a directory")
        entries = []
        for item in sorted(current.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
            if item.name.startswith("."):
                continue
            try:
                is_dir = item.is_dir()
                if not is_dir and item.suffix.lower() != ".csv":
                    continue
                entries.append({
                    "name": item.name,
                    "path": str(item.resolve()),
                    "type": "directory" if is_dir else "csv",
                    "size_bytes": None if is_dir else item.stat().st_size,
                })
            except (OSError, PermissionError):
                continue
        return {
            "roots": [{"name": item.name or str(item), "path": str(item)} for item in self.browse_roots],
            "root": str(root), "path": str(current),
            "parent": str(current.parent) if current != root else None,
            "entries": entries,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "DownstreamStudio/0.1"

    @property
    def app(self) -> StudioApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise ValueError("JSON request is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def send_json(self, value: Any, status: int = 200) -> None:
        content = json.dumps(value, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def error_json(self, error: Exception, status: int = 400) -> None:
        self.send_json({"error": str(error)}, status)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            route = parsed.path
            if route == "/api/health":
                return self.send_json({"ok": True, "workspace": str(self.app.store.root)})
            if route == "/api/projects":
                return self.send_json(self.app.store.list_projects())
            if route == "/api/files":
                query = parse_qs(parsed.query)
                return self.send_json(self.app.browse(query.get("path", [None])[0]))
            match = re.fullmatch(r"/api/projects/([\w-]+)/hpc", route)
            if match:
                project_id = match.group(1)
                if not self.app.store.get_project(project_id):
                    return self.send_json({"error": "Project not found"}, 404)
                profile = self.app.store.get_hpc_connection(project_id) or {
                    "project_id": project_id, "name": "TU Dresden HPC",
                    "host": "login1.barnard.hpc.tu-dresden.de", "port": 22,
                    "username": "", "remote_workspace": "", "last_status": "not_tested",
                    "last_message": "", "last_tested_at": None, "auth_method": "agent",
                    "remote_python": "python3", "worker_version": "",
                }
                saved_password = load_saved_password(project_id, profile["host"], profile["username"])
                if saved_password and project_id not in self.app.hpc_passwords:
                    self.app.hpc_passwords[project_id] = saved_password
                profile["credential_available"] = profile.get("auth_method") == "agent" or project_id in self.app.hpc_passwords
                profile["credential_saved"] = bool(saved_password)
                profile["allowed_hosts"] = sorted(HPC_HOSTS)
                return self.send_json(profile)
            match = re.fullmatch(r"/api/projects/([\w-]+)/hpc/files", route)
            if match:
                query = parse_qs(parsed.query)
                client = self.app.hpc_client(match.group(1))
                try:
                    return self.send_json(browse_remote(client, query.get("path", [None])[0]))
                finally:
                    client.close()
            match = re.fullmatch(r"/api/projects/([\w-]+)", route)
            if match:
                project = self.app.store.get_project(match.group(1))
                if not project:
                    return self.send_json({"error": "Project not found"}, 404)
                project["datasets"] = self.app.store.list_datasets(project["id"])
                project["experiments"] = self.app.store.list_experiments(project["id"])
                return self.send_json(project)
            match = re.fullmatch(r"/api/datasets/([\w-]+)/preview", route)
            if match:
                query = parse_qs(parsed.query)
                return self.preview_dataset(match.group(1), query.get("file", [None])[0])
            match = re.fullmatch(r"/api/experiments/([\w-]+)", route)
            if match:
                self.app.runs.refresh_remote(match.group(1))
                experiment = self.app.store.get_experiment(match.group(1))
                if not experiment:
                    return self.send_json({"error": "Experiment not found"}, 404)
                experiment["log"] = self.app.runs.tail_log(experiment["id"])
                experiment["progress"] = self.app.runs.progress(experiment)
                output = Path(experiment["output_dir"])
                experiment["artifacts"] = sorted(path.name for path in output.iterdir() if path.is_file())
                return self.send_json(experiment)
            match = re.fullmatch(r"/api/experiments/([\w-]+)/results", route)
            if match:
                experiment = self.app.store.get_experiment(match.group(1))
                if not experiment:
                    return self.send_json({"error": "Experiment not found"}, 404)
                output = Path(experiment["output_dir"])
                predictions = read_csv_records(output / "test_predictions.csv", 10001)
                return self.send_json({
                    "metrics": read_csv_records(output / "probe_results.csv"),
                    "best": read_csv_records(output / "best_probe_per_target.csv"),
                    "history": read_csv_records(output / "training_history.csv"),
                    "predictions": predictions[:10000],
                    "predictions_truncated": len(predictions) > 10000,
                })
            match = re.fullmatch(r"/api/experiments/([\w-]+)/artifacts/([^/]+)/preview", route)
            if match:
                return self.preview_artifact(match.group(1), unquote(match.group(2)))
            match = re.fullmatch(r"/api/experiments/([\w-]+)/artifacts/([^/]+)", route)
            if match:
                return self.send_artifact(match.group(1), unquote(match.group(2)))
            return self.send_static(route)
        except Exception as error:
            self.error_json(error, 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            route = urlparse(self.path).path
            if route == "/api/projects":
                body = self.json_body()
                if not body.get("name", "").strip():
                    raise ValueError("Project name is required")
                return self.send_json(self.app.store.create_project(body["name"], body.get("description", "")), 201)
            match = re.fullmatch(r"/api/projects/([\w-]+)/hpc/test", route)
            if match:
                return self.test_hpc_connection(match.group(1))
            match = re.fullmatch(r"/api/projects/([\w-]+)/hpc/deploy", route)
            if match:
                return self.deploy_hpc_worker(match.group(1))
            match = re.fullmatch(r"/api/projects/([\w-]+)/hpc/disconnect", route)
            if match:
                return self.disconnect_hpc(match.group(1))
            match = re.fullmatch(r"/api/projects/([\w-]+)/hpc/forget", route)
            if match:
                return self.forget_hpc_password(match.group(1))
            match = re.fullmatch(r"/api/projects/([\w-]+)/datasets/path", route)
            if match:
                body = self.json_body()
                kind = body["kind"]
                metadata = inspect_dataset(kind, body["path"], body.get("id_column", "id"), body.get("feature_prefix", "f"))
                path = Path(metadata["files"][0]["path"] if kind == "targets" else body["path"]).expanduser()
                dataset = self.app.store.create_dataset(
                    match.group(1), body.get("name") or path.name, kind, "path", path,
                    body.get("id_column", "id"), body.get("feature_prefix", "f"), metadata,
                )
                return self.send_json(dataset, 201)
            match = re.fullmatch(r"/api/projects/([\w-]+)/datasets/hpc", route)
            if match:
                body = self.json_body()
                client = self.app.hpc_client(match.group(1))
                try:
                    metadata = inspect_remote_dataset(
                        client, body["kind"], body["path"], body.get("id_column", "id"), body.get("feature_prefix", "f")
                    )
                finally:
                    client.close()
                dataset = self.app.store.create_dataset(
                    match.group(1), body.get("name") or Path(body["path"]).name, body["kind"], "path", body["path"],
                    body.get("id_column", "id"), body.get("feature_prefix", "f"), metadata,
                )
                return self.send_json(dataset, 201)
            match = re.fullmatch(r"/api/projects/([\w-]+)/datasets/upload", route)
            if match:
                return self.handle_upload(match.group(1))
            match = re.fullmatch(r"/api/datasets/([\w-]+)/transform", route)
            if match:
                dataset = self.app.store.get_dataset(match.group(1))
                if not dataset:
                    return self.send_json({"error": "Dataset not found"}, 404)
                return self.send_json(configure_dataset(self.app.store, dataset, self.json_body()))
            match = re.fullmatch(r"/api/projects/([\w-]+)/experiments", route)
            if match:
                body = self.json_body()
                self.validate_experiment(body)
                experiment = self.app.store.create_experiment(match.group(1), body.get("name", "Experiment"), body)
                self.app.runs.start(experiment["id"])
                return self.send_json(experiment, 201)
            match = re.fullmatch(r"/api/experiments/([\w-]+)/cancel", route)
            if match:
                return self.send_json({"cancelled": self.app.runs.cancel(match.group(1))})
            return self.send_json({"error": "Not found"}, 404)
        except (ValueError, KeyError, FileNotFoundError, ConnectionError) as error:
            self.error_json(error, 400)
        except Exception as error:
            self.error_json(error, 500)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            route = urlparse(self.path).path
            match = re.fullmatch(r"/api/projects/([\w-]+)", route)
            if match:
                project_id = match.group(1)
                profile = self.app.store.get_hpc_connection(project_id)
                project = self.app.store.delete_project(project_id)
                if not project:
                    return self.send_json({"error": "Project not found"}, 404)
                self.app.hpc_passwords.pop(project_id, None)
                credential_warning = None
                if profile:
                    try:
                        forget_saved_password(project_id, profile["host"], profile["username"])
                    except RuntimeError as error:
                        credential_warning = str(error)
                return self.send_json({"deleted": True, "id": project_id, "recoverable_from": project.get("recoverable_from"), "warning": credential_warning})
            match = re.fullmatch(r"/api/datasets/([\w-]+)", route)
            if match:
                dataset = self.app.store.delete_dataset(match.group(1))
                if not dataset:
                    return self.send_json({"error": "Dataset not found"}, 404)
                return self.send_json({"deleted": True, "id": dataset["id"], "recoverable_from": dataset.get("recoverable_from")})
            match = re.fullmatch(r"/api/experiments/([\w-]+)", route)
            if not match:
                return self.send_json({"error": "Not found"}, 404)
            experiment = self.app.store.get_experiment(match.group(1))
            if not experiment:
                return self.send_json({"error": "Experiment not found"}, 404)
            if experiment["status"] in {"queued", "running"}:
                return self.send_json({"error": "Cancel the running experiment before deleting it"}, 409)
            deleted = self.app.store.delete_experiment(experiment["id"])
            return self.send_json({
                "deleted": True,
                "id": experiment["id"],
                "recoverable_from": deleted.get("recoverable_from") if deleted else None,
            })
        except ValueError as error:
            self.error_json(error, 409)
        except Exception as error:
            self.error_json(error, 500)

    def handle_upload(self, project_id: str) -> None:
        kind = self.headers.get("X-Dataset-Kind", "targets")
        filename = Path(unquote(self.headers.get("X-Filename", "data.csv"))).name
        name = unquote(self.headers.get("X-Dataset-Name", filename))
        id_column = self.headers.get("X-Id-Column", "id")
        feature_prefix = self.headers.get("X-Feature-Prefix", "f")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Uploaded file is empty")
        folder = self.app.store.uploads / project_id
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / filename
        temporary = destination.with_suffix(destination.suffix + ".partial")
        remaining = length
        with temporary.open("wb") as output:
            while remaining:
                block = self.rfile.read(min(1024 * 1024, remaining))
                if not block:
                    break
                output.write(block)
                remaining -= len(block)
        if remaining:
            temporary.unlink(missing_ok=True)
            raise ValueError("Upload ended before the advertised file size")
        temporary.replace(destination)
        metadata = inspect_dataset(kind, str(destination), id_column, feature_prefix)
        dataset = self.app.store.create_dataset(
            project_id, name, kind, "upload", destination, id_column, feature_prefix, metadata,
        )
        self.send_json(dataset, 201)

    def test_hpc_connection(self, project_id: str) -> None:
        if not self.app.store.get_project(project_id):
            return self.send_json({"error": "Project not found"}, 404)
        body = self.json_body()
        host = str(body.get("host", "")).strip()
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        auth_method = str(body.get("auth_method", "agent"))
        port = int(body.get("port", 22))
        if port < 1 or port > 65535:
            raise ValueError("SSH port must be between 1 and 65535")
        profile = {
            "name": "TU Dresden HPC", "host": host, "port": port, "username": username,
            "remote_workspace": "~/.downstream-studio", "auth_method": auth_method,
            "remote_python": str(body.get("remote_python", "python3")).strip() or "python3",
            "worker_version": str(body.get("worker_version", "")),
            "last_status": "testing", "last_message": "Testing SSH authentication", "last_tested_at": now(),
        }
        if auth_method == "password" and not password:
            password = self.app.hpc_passwords.get(project_id) or load_saved_password(project_id, host, username) or ""
        if auth_method == "password" and not password:
            raise ValueError("Enter your ZIH password or save one in the operating-system credential store")
        self.app.store.save_hpc_connection(project_id, profile)
        try:
            details = test_connection(host, port, username, password, auth_method)
        except Exception as error:
            profile.update(last_status="failed", last_message=str(error), last_tested_at=now())
            self.app.store.save_hpc_connection(project_id, profile)
            raise
        if password:
            self.app.hpc_passwords[project_id] = password
        credential_saved = bool(load_saved_password(project_id, host, username))
        save_warning = ""
        if auth_method == "password" and body.get("save_password", True):
            try:
                store_saved_password(project_id, host, username, password)
                credential_saved = True
            except RuntimeError as error:
                save_warning = str(error)
        profile.update(last_status="connected", last_message=f"Connected to {details['hostname']}", last_tested_at=now())
        saved = self.app.store.save_hpc_connection(project_id, profile)
        client = self.app.hpc_client(project_id)
        try:
            installed = worker_status(client)
        finally:
            client.close()
        if installed.get("version") and installed["version"] != saved.get("worker_version"):
            saved["worker_version"] = installed["version"]
            saved = self.app.store.save_hpc_connection(project_id, saved)
        return self.send_json({**saved, **details, **installed, "credential_available": True,
                               "credential_saved": credential_saved, "save_warning": save_warning})

    def deploy_hpc_worker(self, project_id: str) -> None:
        profile = self.app.store.get_hpc_connection(project_id)
        if not profile:
            return self.send_json({"error": "Connect to the HPC before deploying the worker"}, 409)
        client = self.app.hpc_client(project_id)
        try:
            deployed = deploy_worker(client, profile.get("remote_python", "python3"))
        finally:
            client.close()
        profile["worker_version"] = deployed["version"]
        profile["last_message"] = f"Worker {deployed['version']} deployed"
        saved = self.app.store.save_hpc_connection(project_id, profile)
        return self.send_json({**saved, **deployed, "installed": True})

    def disconnect_hpc(self, project_id: str) -> None:
        profile = self.app.store.get_hpc_connection(project_id)
        if not profile:
            return self.send_json({"error": "HPC connection is not configured"}, 404)
        self.app.hpc_passwords.pop(project_id, None)
        profile.update(last_status="disconnected", last_message="Session credential removed", last_tested_at=now())
        saved = self.app.store.save_hpc_connection(project_id, profile)
        return self.send_json({**saved, "credential_available": False, "credential_saved": bool(load_saved_password(project_id, profile["host"], profile["username"]))})

    def forget_hpc_password(self, project_id: str) -> None:
        profile = self.app.store.get_hpc_connection(project_id)
        if not profile:
            return self.send_json({"error": "HPC connection is not configured"}, 404)
        self.app.hpc_passwords.pop(project_id, None)
        forget_saved_password(project_id, profile["host"], profile["username"])
        profile.update(last_status="disconnected", last_message="Saved credential removed", last_tested_at=now())
        saved = self.app.store.save_hpc_connection(project_id, profile)
        return self.send_json({**saved, "credential_available": False, "credential_saved": False})

    def validate_experiment(self, config: Dict[str, Any]) -> None:
        embeddings = self.app.store.get_dataset(config.get("embeddings_dataset_id", ""))
        targets = self.app.store.get_dataset(config.get("targets_dataset_id", ""))
        if not embeddings or embeddings["kind"] != "embeddings":
            raise ValueError("Select a valid embeddings dataset")
        if not targets or targets["kind"] != "targets":
            raise ValueError("Select a valid targets dataset")
        if embeddings["project_id"] != targets["project_id"]:
            raise ValueError("Datasets must belong to the same project")
        execution_target = config.get("execution_target", "local")
        embedding_remote = embeddings.get("metadata", {}).get("location") == "hpc"
        target_remote = targets.get("metadata", {}).get("location") == "hpc"
        if execution_target == "hpc":
            if not embedding_remote or not target_remote:
                raise ValueError("HPC experiments require both datasets to be registered from HPC storage")
            if embeddings.get("metadata", {}).get("view") or targets.get("metadata", {}).get("view"):
                raise ValueError("Edited remote dataset views are not supported yet; register the original HPC files")
            profile = self.app.store.get_hpc_connection(embeddings["project_id"])
            if not profile or not profile.get("worker_version"):
                raise ValueError("Connect to the HPC and deploy the worker before starting remote training")
        elif execution_target == "local" and (embedding_remote or target_remote):
            raise ValueError("Remote datasets must be trained with the HPC execution target")
        else:
            if execution_target not in {"local", "hpc"}:
                raise ValueError("Execution target must be local or HPC")
        selected = config.get("target_columns", [])
        available = set(targets["metadata"].get("target_columns", []))
        if not selected or not set(selected) <= available:
            raise ValueError("Select one or more available numeric targets")
        probes = set(config.get("probes", []))
        if not probes or not probes <= {"linear", "mlp"}:
            raise ValueError("Probe must be linear and/or mlp")
        if not 0 <= int(config.get("linear_width", 0)) <= 8192:
            raise ValueError("Linear bottleneck width must be between 0 and 8192")
        layers = config.get("layers", [])
        allowed_activations = {"relu", "gelu", "silu", "tanh", "leaky_relu", "identity"}
        if "mlp" in probes:
            if not layers or len(layers) > 12:
                raise ValueError("An MLP requires between 1 and 12 hidden layers")
            if any(int(layer.get("units", 0)) < 1 or int(layer.get("units", 0)) > 8192 for layer in layers):
                raise ValueError("Each hidden layer must contain between 1 and 8192 neurons")
            if any(layer.get("activation") not in allowed_activations for layer in layers):
                raise ValueError("An MLP layer uses an unsupported activation")
        fractions = config.get("train_fractions", [])
        epochs = config.get("epochs", [])
        if not fractions or any(not 0 < float(x) < 1 for x in fractions):
            raise ValueError("Training fractions must be between 0 and 1")
        if not epochs or any(int(x) < 1 for x in epochs):
            raise ValueError("Epochs must be positive")

    def send_artifact(self, experiment_id: str, name: str) -> None:
        experiment = self.app.store.get_experiment(experiment_id)
        if not experiment or Path(name).name != name:
            return self.send_json({"error": "Artifact not found"}, 404)
        path = Path(experiment["output_dir"]) / name
        if not path.is_file():
            return self.send_json({"error": "Artifact not found"}, 404)
        self.send_file(path, download=True)

    def preview_dataset(self, dataset_id: str, requested_file: str | None) -> None:
        dataset = self.app.store.get_dataset(dataset_id)
        if not dataset:
            return self.send_json({"error": "Dataset not found"}, 404)
        if dataset.get("metadata", {}).get("location") == "hpc":
            files = dataset["metadata"].get("files", [])
            selected = requested_file or (files[0]["path"] if files else dataset["path"])
            if selected not in {item["path"] for item in files}:
                return self.send_json({"error": "Dataset file not found"}, 404)
            client = self.app.hpc_client(dataset["project_id"])
            try:
                preview = preview_remote_csv(client, selected)
            finally:
                client.close()
            preview.update({
                "name": dataset["name"], "files": [{"name": item["filename"], "path": item["path"],
                "size_bytes": item["size_bytes"]} for item in files],
                "editor_columns": preview["columns"], "editor_selected": preview["columns"],
                "editor_renames": {}, "editor_id_column": dataset["id_column"], "editor_rows": preview["rows"],
                "remote": True,
            })
            return self.send_json(preview)
        files = [Path(item["path"]).expanduser().resolve() for item in dataset.get("metadata", {}).get("files", [])]
        files = [path for path in files if path.is_file()]
        if not files:
            path = Path(dataset["path"]).expanduser().resolve()
            files = [path] if path.is_file() else []
        if not files:
            return self.send_json({"error": "No readable CSV files were found for this dataset"}, 404)
        selected = files[0]
        if requested_file:
            matches = [path for path in files if str(path) == requested_file]
            if not matches:
                return self.send_json({"error": "Dataset file not found"}, 404)
            selected = matches[0]
        raw_records = read_csv_records(selected, 501)
        metadata = dataset.get("metadata", {})
        source_metadata = metadata.get("source_metadata") or metadata
        source_columns = list(source_metadata.get("files", [{}])[0].get("columns", []))
        view = metadata.get("view") or {"columns": source_columns, "renames": {}}
        active_sources = list(view.get("columns", source_columns))
        active_renames = dict(view.get("renames", {}))
        id_source = next((column for column in source_columns if active_renames.get(column, column) == dataset["id_column"]), dataset["id_column"])
        columns, records = apply_dataset_view(dataset, raw_records)
        return self.send_json({
            "name": dataset["name"], "file": str(selected), "size_bytes": selected.stat().st_size,
            "type": "csv", "columns": columns, "rows": records[:500],
            "truncated": len(records) > 500,
            "files": [{"name": path.name, "path": str(path), "size_bytes": path.stat().st_size} for path in files],
            "editor_columns": source_columns, "editor_selected": active_sources,
            "editor_renames": active_renames, "editor_id_column": id_source,
            "editor_rows": raw_records[:500],
        })

    def preview_artifact(self, experiment_id: str, name: str) -> None:
        experiment = self.app.store.get_experiment(experiment_id)
        if not experiment or Path(name).name != name:
            return self.send_json({"error": "Artifact not found"}, 404)
        path = Path(experiment["output_dir"]) / name
        if not path.is_file():
            return self.send_json({"error": "Artifact not found"}, 404)
        suffix = path.suffix.lower()
        base = {"name": name, "size_bytes": path.stat().st_size}
        if suffix == ".csv":
            records = read_csv_records(path, 501)
            return self.send_json({
                **base, "type": "csv", "columns": list(records[0]) if records else [],
                "rows": records[:500], "truncated": len(records) > 500,
            })
        if suffix == ".json":
            if path.stat().st_size > 2_000_000:
                raise ValueError("JSON artifact is too large for an in-browser preview")
            with path.open(encoding="utf-8") as handle:
                return self.send_json({**base, "type": "json", "value": json.load(handle)})
        if suffix in {".log", ".txt"}:
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 250_000))
                text = handle.read().decode("utf-8", errors="replace")
            lines = text.splitlines()
            return self.send_json({
                **base, "type": "log", "lines": lines[-1000:],
                "truncated": path.stat().st_size > 250_000 or len(lines) > 1000,
            })
        return self.send_json({"error": "This artifact type can only be downloaded"}, 415)

    def send_static(self, route: str) -> None:
        requested = "index.html" if route in {"/", ""} else route.lstrip("/")
        path = (self.app.web / requested).resolve()
        if self.app.web.resolve() not in path.parents or not path.is_file():
            path = self.app.web / "index.html"
        self.send_file(path)

    def send_file(self, path: Path, download: bool = False) -> None:
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Downstream Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", default="~/.downstream-studio")
    parser.add_argument(
        "--browse-root", action="append", default=[],
        help="Directory users may browse when selecting files (repeatable; defaults to current directory)",
    )
    args = parser.parse_args()
    app = StudioApp(
        Path(args.workspace).expanduser(),
        [Path(path) for path in args.browse_root] if args.browse_root else None,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app  # type: ignore[attr-defined]
    print(f"Downstream Studio: http://{args.host}:{args.port}")
    print(f"Workspace: {app.store.root}")
    print("Browse roots: " + ", ".join(map(str, app.browse_roots)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
