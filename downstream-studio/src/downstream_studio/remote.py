"""Outbound-only SSH/SFTP deployment and Slurm execution."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import io
import json
import os
import posixpath
import re
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict

import paramiko


REMOTE_FOLDER = ".downstream-studio"
SUMMARY_ARTIFACTS = {
    "probe_results.csv", "best_probe_per_target.csv", "training_history.csv",
    "studio_config.json", "command.json", "run.log",
}


def _checked(client: paramiko.SSHClient, command: str, timeout: int = 60) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace").strip()
    if code:
        raise RuntimeError(error or output.strip() or f"Remote command failed with status {code}")
    return output


def _mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    current = "/" if path.startswith("/") else ""
    for part in path.strip("/").split("/"):
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current, mode=0o700)


def remote_layout(sftp: paramiko.SFTPClient) -> Dict[str, str]:
    home = sftp.normalize(".")
    root = posixpath.join(home, REMOTE_FOLDER)
    return {
        "home": home,
        "root": root,
        "worker": posixpath.join(root, "worker"),
        "venv": posixpath.join(root, "worker", "venv"),
        "jobs": posixpath.join(root, "jobs"),
        "results": posixpath.join(root, "results"),
    }


def _source_files(source_root: Path) -> list[Path]:
    allowed = {".py", ".toml", ".md"}
    return sorted(path for path in source_root.rglob("*") if path.is_file() and path.suffix in allowed
                  and not any(part.startswith(".") or part in {"__pycache__", "node_modules"} for part in path.parts))


def build_worker_bundle() -> tuple[Path, str]:
    repository = Path(__file__).resolve().parents[3]
    analysis = repository / "downstream-analysis"
    if not (analysis / "pyproject.toml").is_file():
        raise FileNotFoundError("The downstream-analysis source package is not available beside Studio")
    files = _source_files(analysis)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(repository)).encode())
        digest.update(path.read_bytes())
    version = digest.hexdigest()[:12]
    descriptor, name = tempfile.mkstemp(prefix="downstream-worker-", suffix=".tar.gz")
    os.close(descriptor)
    temporary = Path(name)
    with tarfile.open(temporary, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(repository)))
    return temporary, version


def deploy_worker(client: paramiko.SSHClient, remote_python: str = "python3") -> Dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_./+-]+", remote_python):
        raise ValueError("Remote Python must be a command or absolute path without shell syntax")
    archive, version = build_worker_bundle()
    try:
        sftp = client.open_sftp()
        layout = remote_layout(sftp)
        release = posixpath.join(layout["worker"], "releases", version)
        for folder in (layout["root"], layout["worker"], posixpath.dirname(release), layout["jobs"], layout["results"]):
            _mkdirs(sftp, folder)
        remote_archive = posixpath.join(layout["worker"], f"worker-{version}.tar.gz")
        sftp.put(str(archive), remote_archive)
        sftp.chmod(remote_archive, 0o600)
        sftp.close()
        command = " && ".join([
            f"mkdir -p -- {shlex.quote(release)}",
            f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(release)}",
            f"test -x {shlex.quote(posixpath.join(layout['venv'], 'bin', 'python'))} || {shlex.quote(remote_python)} -m venv {shlex.quote(layout['venv'])}",
            f"{shlex.quote(posixpath.join(layout['venv'], 'bin', 'python'))} -m pip install --disable-pip-version-check {shlex.quote(posixpath.join(release, 'downstream-analysis') + '[csv]')}",
            f"printf %s {shlex.quote(version)} > {shlex.quote(posixpath.join(layout['worker'], 'CURRENT'))}",
        ])
        output = _checked(client, command, timeout=1800)
        return {"version": version, "remote_root": layout["root"], "release": release, "output": output[-4000:]}
    finally:
        archive.unlink(missing_ok=True)


def worker_status(client: paramiko.SSHClient) -> Dict[str, Any]:
    sftp = client.open_sftp()
    layout = remote_layout(sftp)
    current = posixpath.join(layout["worker"], "CURRENT")
    try:
        with sftp.open(current, "r") as handle:
            version = handle.read().decode().strip()
    except OSError:
        version = ""
    sftp.close()
    return {"installed": bool(version), "version": version, "remote_root": layout["root"]}


def browse_remote(client: paramiko.SSHClient, requested: str | None) -> Dict[str, Any]:
    sftp = client.open_sftp()
    home = sftp.normalize(".")
    current = home if not requested or requested == "~" else requested
    if not current.startswith("/"):
        raise ValueError("Remote paths must be absolute or ~")
    entries = []
    for item in sorted(sftp.listdir_attr(current), key=lambda value: value.filename.lower()):
        if item.filename.startswith("."):
            continue
        is_dir = bool(item.st_mode & 0o040000)
        if not is_dir and not item.filename.lower().endswith(".csv"):
            continue
        entries.append({
            "name": item.filename, "path": posixpath.join(current, item.filename),
            "type": "directory" if is_dir else "csv", "size_bytes": None if is_dir else item.st_size,
        })
    parent = posixpath.dirname(current.rstrip("/"))
    sftp.close()
    return {"roots": [{"name": "Remote home", "path": home}], "path": current,
            "parent": parent if current != "/" else None, "entries": entries, "remote": True}


def _remote_csv_paths(sftp: paramiko.SFTPClient, value: str, kind: str) -> list[str]:
    try:
        stat = sftp.stat(value)
    except OSError as error:
        raise FileNotFoundError(f"Remote path is not readable: {value}") from error
    if not bool(stat.st_mode & 0o040000):
        return [value]
    if kind != "embeddings":
        raise ValueError("A target dataset must point to one CSV file")
    found: list[str] = []
    pending = [(value, 0)]
    while pending and len(found) < 100:
        folder, depth = pending.pop(0)
        for item in sftp.listdir_attr(folder):
            path = posixpath.join(folder, item.filename)
            if bool(item.st_mode & 0o040000) and depth < 3:
                pending.append((path, depth + 1))
            elif fnmatch.fnmatch(item.filename, "embeddings*.csv"):
                found.append(path)
    if not found:
        raise FileNotFoundError(f"No embeddings*.csv files found under {value}")
    return sorted(found)


def _inspect_remote_csv(sftp: paramiko.SFTPClient, path: str, id_column: str) -> Dict[str, Any]:
    stat = sftp.stat(path)
    with sftp.open(path, "r") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        columns = reader.fieldnames or []
        if id_column not in columns:
            raise ValueError(f"Missing ID column {id_column!r} in {path}")
        numeric = {column: True for column in columns if column != id_column}
        nonempty = {column: 0 for column in columns if column != id_column}
        examples, count = [], 0
        for row in reader:
            count += 1
            if len(examples) < 3:
                examples.append({key: row.get(key, "") for key in columns[:8]})
            for column in numeric:
                value = (row.get(column) or "").strip()
                if value:
                    nonempty[column] += 1
                    try:
                        float(value)
                    except ValueError:
                        numeric[column] = False
            if count >= 500:
                break
    numeric_columns = [column for column in columns if numeric.get(column) and nonempty.get(column, 0)]
    return {"path": path, "filename": posixpath.basename(path), "size_bytes": stat.st_size,
            "columns": columns, "numeric_columns": numeric_columns, "sampled_rows": count,
            "at_least_rows": count, "examples": examples}


def inspect_remote_dataset(client: paramiko.SSHClient, kind: str, value: str,
                           id_column: str = "id", feature_prefix: str = "f") -> Dict[str, Any]:
    sftp = client.open_sftp()
    paths = _remote_csv_paths(sftp, value, kind)
    files = [_inspect_remote_csv(sftp, path, id_column) for path in paths]
    sftp.close()
    result: Dict[str, Any] = {"kind": kind, "files": files, "file_count": len(files), "location": "hpc"}
    if kind == "embeddings":
        for item in files:
            item["feature_columns"] = [column for column in item["numeric_columns"] if column.startswith(feature_prefix)]
            item["embedding_dim"] = len(item["feature_columns"])
            item["model_name"] = posixpath.basename(posixpath.dirname(item["path"]))
        result["models"] = [item["model_name"] for item in files]
        result["dimensions"] = sorted({item["embedding_dim"] for item in files})
    else:
        result["target_columns"] = files[0]["numeric_columns"]
    return result


def preview_remote_csv(client: paramiko.SSHClient, path: str, limit: int = 500) -> Dict[str, Any]:
    sftp = client.open_sftp()
    stat = sftp.stat(path)
    with sftp.open(path, "r") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        columns = reader.fieldnames or []
        rows = []
        for row in reader:
            rows.append(dict(row))
            if len(rows) > limit:
                break
    sftp.close()
    return {"file": path, "size_bytes": stat.st_size, "type": "csv", "columns": columns,
            "rows": rows[:limit], "truncated": len(rows) > limit}


def submit_job(client: paramiko.SSHClient, experiment: Dict[str, Any], embeddings: Dict[str, Any],
               targets: Dict[str, Any]) -> Dict[str, str]:
    config = experiment["config"]
    resources = config.get("slurm", {})
    partition = str(resources.get("partition", "barnard"))
    if partition not in {"barnard", "capella"}:
        raise ValueError("Slurm partition must be barnard or capella")
    cpus = max(1, min(64, int(resources.get("cpus", 4))))
    memory = max(1, min(512, int(resources.get("memory_gb", 16))))
    hours = max(1, min(168, int(resources.get("hours", 8))))
    gpus = max(0, min(8, int(resources.get("gpus", 0))))
    if partition == "barnard" and gpus:
        raise ValueError("Barnard jobs cannot request GPUs")
    sftp = client.open_sftp()
    layout = remote_layout(sftp)
    status = worker_status(client)
    if not status["installed"]:
        raise RuntimeError("Deploy the HPC worker before submitting an experiment")
    job_dir = posixpath.join(layout["jobs"], experiment["id"])
    output_dir = posixpath.join(layout["results"], experiment["id"])
    _mkdirs(sftp, job_dir)
    _mkdirs(sftp, output_dir)
    selected = set(config.get("embedding_models") or [])
    embedding_paths = [item["path"] for item in embeddings["metadata"]["files"]
                       if not selected or item.get("model_name") in selected]
    payload = {**config, "embeddings": embedding_paths,
               "targets_csv": targets["metadata"]["files"][0]["path"], "output_dir": output_dir}
    config_path = posixpath.join(job_dir, "config.json")
    script_path = posixpath.join(job_dir, "job.sbatch")
    with sftp.open(config_path, "w") as handle:
        handle.write(json.dumps(payload, indent=2))
    python = posixpath.join(layout["venv"], "bin", "python")
    lines = [
        "#!/bin/bash", f"#SBATCH --job-name=ds-{experiment['id']}", f"#SBATCH --partition={partition}",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}G", f"#SBATCH --time={hours:02d}:00:00",
        f"#SBATCH --output={output_dir}/run.log", f"#SBATCH --error={output_dir}/run.log",
    ]
    account = str(resources.get("account", "")).strip()
    if account:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", account):
            raise ValueError("Invalid Slurm account")
        lines.append(f"#SBATCH --account={account}")
    if gpus:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    lines.extend(["set -euo pipefail", f"export OMP_NUM_THREADS={cpus}",
                  f"exec {shlex.quote(python)} -m interface_downstream.remote_worker {shlex.quote(config_path)}"])
    with sftp.open(script_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    sftp.chmod(script_path, 0o700)
    sftp.close()
    job_id = _checked(client, f"sbatch --parsable {shlex.quote(script_path)}").strip().split(";")[0]
    if not re.fullmatch(r"\d+", job_id):
        raise RuntimeError(f"Slurm returned an unexpected job id: {job_id}")
    return {"job_id": job_id, "remote_output_dir": output_dir}


def job_state(client: paramiko.SSHClient, job_id: str) -> Dict[str, Any]:
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError("Invalid Slurm job id")
    queued = _checked(client, f"squeue -h -j {job_id} -o %T || true").strip().splitlines()
    if queued:
        state = queued[0].upper()
        return {"state": state, "status": "queued" if state == "PENDING" else "running"}
    accounting = _checked(client, f"sacct -n -P -X -j {job_id} --format=State,ExitCode || true").strip().splitlines()
    state, exit_code = (accounting[0].split("|") + [""])[:2] if accounting else ("UNKNOWN", "")
    normalized = state.split()[0].split("+")[0].upper()
    if normalized == "COMPLETED":
        status = "completed"
    elif normalized in {"CANCELLED"}:
        status = "cancelled"
    elif normalized in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}:
        status = "failed"
    else:
        status = "running"
    return {"state": normalized, "status": status, "exit_code": exit_code}


def sync_results(client: paramiko.SSHClient, remote_output: str, local_output: Path,
                 include_predictions: bool = False) -> list[str]:
    sftp = client.open_sftp()
    local_output.mkdir(parents=True, exist_ok=True)
    names = set(SUMMARY_ARTIFACTS)
    if include_predictions:
        names.add("test_predictions.csv")
    downloaded = []
    for name in names:
        remote = posixpath.join(remote_output, name)
        try:
            stat = sftp.stat(remote)
        except OSError:
            continue
        limit = 100_000_000 if name == "test_predictions.csv" else 25_000_000
        if stat.st_size > limit:
            continue
        temporary = local_output / f".{name}.partial"
        sftp.get(remote, str(temporary))
        temporary.replace(local_output / name)
        downloaded.append(name)
    sftp.close()
    return sorted(downloaded)


def cancel_job(client: paramiko.SSHClient, job_id: str) -> None:
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError("Invalid Slurm job id")
    _checked(client, f"scancel {job_id}")
