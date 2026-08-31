"""Secure SSH connection checks for HPC execution backends."""

from __future__ import annotations

import base64
import codecs
import hashlib
import shlex
import socket
import threading
import time
import uuid
from typing import Any, Dict, Optional

import paramiko
import keyring
from keyring.errors import KeyringError, PasswordDeleteError


CAPELLA_HOSTS = {
    "login1.capella.hpc.tu-dresden.de",
    "login2.capella.hpc.tu-dresden.de",
}
CAPELLA_HOST_KEY_FINGERPRINTS = {
    "SHA256:wc0g+OaX+4XzVvfSk0OwXu92kKxsBh+9ou4ErTl8Omg",  # RSA
    "SHA256:zZiM3YZ0HBK6nxBS5VuJUwudmrDn/A7ajU1ec6vgVkU",  # ED25519
    "SHA256:1bQ0RhIIwvl+GpsW53KncPn+2z/f149WAwK5wipEgx8",  # ECDSA
}
KEYRING_SERVICE = "Downstream Studio · HPC"


def credential_account(project_id: str, host: str, username: str) -> str:
    return f"{project_id}:{username}@{host}"


def load_password(project_id: str, host: str, username: str) -> Optional[str]:
    if not username:
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, credential_account(project_id, host, username))
    except KeyringError:
        return None


def save_password(project_id: str, host: str, username: str, password: str) -> None:
    try:
        keyring.set_password(KEYRING_SERVICE, credential_account(project_id, host, username), password)
    except KeyringError as error:
        raise RuntimeError(f"The operating-system credential store could not save the password: {error}") from error


def forget_password(project_id: str, host: str, username: str) -> None:
    try:
        keyring.delete_password(KEYRING_SERVICE, credential_account(project_id, host, username))
    except PasswordDeleteError:
        pass
    except KeyringError as error:
        raise RuntimeError(f"The operating-system credential store could not remove the password: {error}") from error


def sha256_fingerprint(key: paramiko.PKey) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


class PublishedCapellaHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client: paramiko.SSHClient, hostname: str, key: paramiko.PKey) -> None:
        if hostname not in CAPELLA_HOSTS or sha256_fingerprint(key) not in CAPELLA_HOST_KEY_FINGERPRINTS:
            raise paramiko.SSHException("The SSH host key does not match TU Dresden's published Capella fingerprint")
        client.get_host_keys().add(hostname, key.get_name(), key)


def connect_client(host: str, port: int, username: str, password: str) -> paramiko.SSHClient:
    if host not in CAPELLA_HOSTS:
        raise ValueError("Choose one of the official Capella login nodes")
    if not username.strip() or not password:
        raise ValueError("ZIH username and password are required")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(PublishedCapellaHostKeyPolicy())
    try:
        client.connect(
            hostname=host, port=port, username=username.strip(), password=password,
            look_for_keys=False, allow_agent=False, timeout=12, auth_timeout=20, banner_timeout=12,
        )
        return client
    except (paramiko.AuthenticationException, paramiko.BadAuthenticationType) as error:
        client.close()
        raise ValueError("Authentication failed. Check your ZIH username, password, and VPN connection.") from error
    except (socket.timeout, socket.gaierror, paramiko.SSHException, OSError) as error:
        client.close()
        raise ConnectionError(f"Could not establish the Capella SSH connection: {error}") from error


def test_connection(host: str, port: int, username: str, password: str) -> Dict[str, Any]:
    client = connect_client(host, port, username, password)
    try:
        _, stdout, stderr = client.exec_command("hostname; id -un; command -v sbatch >/dev/null && echo slurm-ready", timeout=12)
        output = stdout.read().decode("utf-8", errors="replace").splitlines()
        error = stderr.read().decode("utf-8", errors="replace").strip()
        if error and not output:
            raise RuntimeError(error)
        return {
            "hostname": output[0] if output else host,
            "remote_user": output[1] if len(output) > 1 else username,
            "slurm_available": "slurm-ready" in output,
            "host_key_verified": True,
        }
    finally:
        client.close()


class TerminalSession:
    """One bounded interactive SSH pseudo-terminal."""

    max_output = 1_000_000

    def __init__(self, client: paramiko.SSHClient, remote_workspace: str = "", cols: int = 120, rows: int = 32):
        self.id = uuid.uuid4().hex[:16]
        self.client = client
        self.channel = client.invoke_shell(term="xterm-256color", width=cols, height=rows)
        self.output = ""
        self.base_cursor = 0
        self.lock = threading.Lock()
        self.closed = False
        self.last_activity = time.time()
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        if remote_workspace:
            self.channel.send(f"cd -- {shlex.quote(remote_workspace)}\n")
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def _read_loop(self) -> None:
        try:
            while not self.channel.closed:
                if self.channel.recv_ready():
                    text = self.decoder.decode(self.channel.recv(32768))
                    with self.lock:
                        self.output += text
                        if len(self.output) > self.max_output:
                            removed = len(self.output) - self.max_output
                            self.output = self.output[removed:]
                            self.base_cursor += removed
                    self.last_activity = time.time()
                else:
                    time.sleep(0.03)
        finally:
            self.closed = True

    def read(self, cursor: int) -> Dict[str, Any]:
        with self.lock:
            start = max(0, cursor - self.base_cursor)
            data = self.output[start:]
            next_cursor = self.base_cursor + len(self.output)
        return {"data": data, "cursor": next_cursor, "closed": self.closed or self.channel.closed}

    def write(self, data: str) -> None:
        if self.closed or self.channel.closed:
            raise ConnectionError("The terminal session is closed")
        self.channel.send(data)
        self.last_activity = time.time()

    def resize(self, cols: int, rows: int) -> None:
        self.channel.resize_pty(width=max(20, min(cols, 500)), height=max(5, min(rows, 200)))

    def close(self) -> None:
        self.closed = True
        self.channel.close()
        self.client.close()
