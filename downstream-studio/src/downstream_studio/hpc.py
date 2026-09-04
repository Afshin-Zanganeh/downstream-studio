"""Secure SSH connection checks for HPC execution backends."""

from __future__ import annotations

import base64
import hashlib
import socket
from typing import Any, Dict, Optional

import paramiko
import keyring
from keyring.errors import KeyringError, PasswordDeleteError


CAPELLA_HOSTS = {
    "login1.capella.hpc.tu-dresden.de",
    "login2.capella.hpc.tu-dresden.de",
}
BARNARD_HOSTS = {
    "login1.barnard.hpc.tu-dresden.de",
    "login2.barnard.hpc.tu-dresden.de",
    "login3.barnard.hpc.tu-dresden.de",
    "login4.barnard.hpc.tu-dresden.de",
}
CAPELLA_HOST_KEY_FINGERPRINTS = {
    "SHA256:wc0g+OaX+4XzVvfSk0OwXu92kKxsBh+9ou4ErTl8Omg",  # RSA
    "SHA256:zZiM3YZ0HBK6nxBS5VuJUwudmrDn/A7ajU1ec6vgVkU",  # ED25519
    "SHA256:1bQ0RhIIwvl+GpsW53KncPn+2z/f149WAwK5wipEgx8",  # ECDSA
}
BARNARD_HOST_KEY_FINGERPRINTS = {
    "SHA256:lVQOvnci07jkxmFnX58pQf3cD7lz1mf4K4b9jZrAlVU",  # RSA
    "SHA256:Gn4n5IX9eEvkpOGrtZzs9T9yAfJUB200bgRchchiKAQ",  # ED25519
    "SHA256:Xan2MYazewT0V5agNazaQfWzLKBD3P48zRwR6reoXhI",  # ECDSA
}
HPC_HOSTS = CAPELLA_HOSTS | BARNARD_HOSTS
PUBLISHED_HOST_KEYS = {
    **{host: CAPELLA_HOST_KEY_FINGERPRINTS for host in CAPELLA_HOSTS},
    **{host: BARNARD_HOST_KEY_FINGERPRINTS for host in BARNARD_HOSTS},
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


class PublishedHpcHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client: paramiko.SSHClient, hostname: str, key: paramiko.PKey) -> None:
        if hostname not in HPC_HOSTS or sha256_fingerprint(key) not in PUBLISHED_HOST_KEYS[hostname]:
            raise paramiko.SSHException("The SSH host key does not match TU Dresden's published fingerprint")
        client.get_host_keys().add(hostname, key.get_name(), key)


def connect_client(
    host: str, port: int, username: str, password: str = "", auth_method: str = "agent"
) -> paramiko.SSHClient:
    if host not in HPC_HOSTS:
        raise ValueError("Choose one of the approved TU Dresden login nodes")
    if not username.strip():
        raise ValueError("A ZIH username is required")
    if auth_method not in {"agent", "password"}:
        raise ValueError("Authentication must use SSH agent or password")
    if auth_method == "password" and not password:
        raise ValueError("A ZIH password is required for password authentication")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(PublishedHpcHostKeyPolicy())
    try:
        client.connect(
            hostname=host, port=port, username=username.strip(),
            password=password if auth_method == "password" else None,
            look_for_keys=auth_method == "agent", allow_agent=auth_method == "agent",
            timeout=12, auth_timeout=20, banner_timeout=12,
        )
        return client
    except (paramiko.AuthenticationException, paramiko.BadAuthenticationType) as error:
        client.close()
        detail = "SSH agent/key" if auth_method == "agent" else "password"
        raise ValueError(f"Authentication failed. Check your ZIH username, {detail}, and VPN connection.") from error
    except (socket.timeout, socket.gaierror, paramiko.SSHException, OSError) as error:
        client.close()
        raise ConnectionError(f"Could not establish the HPC SSH connection: {error}") from error


def test_connection(
    host: str, port: int, username: str, password: str = "", auth_method: str = "agent"
) -> Dict[str, Any]:
    client = connect_client(host, port, username, password, auth_method)
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
