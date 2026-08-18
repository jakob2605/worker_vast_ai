"""Direct, resumable server-to-server library migration.

The source worker packages the library into one temporary tar chunk at a time
and uploads each chunk to the destination over an rclone SFTP remote.  The
destination extracts and deletes each chunk before the next one is sent.  No
Google Drive object or permanent archive is involved.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import requests

from .config import DB_PATH, LIBRARY_DIR

Progress = Callable[[str, float], None]
MIGRATION_FORMAT = "vastai-library-migration-v1"


class MigrationError(RuntimeError):
    pass


def _safe_members(tar: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise MigrationError(f"Unsafe migration archive member: {member.name}")
        if member.issym() or member.islnk():
            raise MigrationError(f"Links are not allowed in migration archives: {member.name}")


def _sqlite_snapshot(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as source, sqlite3.connect(target) as copy:
        source.backup(copy)
        integrity = copy.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise MigrationError(f"Source SQLite integrity check failed: {integrity}")


def _files_for_migration(
    include_movies: bool,
    include_clips: bool,
    include_frames: bool,
    include_embeddings: bool,
    include_metadata: bool,
    db_copy: Path,
) -> list[tuple[Path, str]]:
    folders = [
        ("movies", include_movies),
        ("clips", include_clips),
        ("frames", include_frames),
        ("embeddings", include_embeddings),
        ("metadata", include_metadata),
    ]
    files: list[tuple[Path, str]] = [(db_copy, "movie_clips.sqlite3")]
    links = LIBRARY_DIR / "download_links.jsonl"
    if links.is_file():
        files.append((links, "download_links.jsonl"))
    for folder, enabled in folders:
        if not enabled:
            continue
        source = LIBRARY_DIR / folder
        if not source.exists():
            continue
        files.extend(
            (path, str(Path(folder) / path.relative_to(source)))
            for path in sorted(source.rglob("*"))
            if path.is_file() and not path.is_symlink()
        )
    return files


def _plan_chunks(files: list[tuple[Path, str]], chunk_bytes: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[tuple[Path, str]] = []
    current_bytes = 0
    for path, arcname in files:
        size = path.stat().st_size
        # A file is never split.  A single file may therefore exceed the limit.
        if current and current_bytes + size > chunk_bytes:
            chunks.append({"name": f"chunk-{len(chunks):06d}.tar", "bytes": current_bytes, "files": [name for _, name in current]})
            current = []
            current_bytes = 0
        current.append((path, arcname))
        current_bytes += size
    if current:
        chunks.append({"name": f"chunk-{len(chunks):06d}.tar", "bytes": current_bytes, "files": [name for _, name in current]})
    return chunks


def _make_chunk(files_by_name: dict[str, Path], names: list[str], output: Path) -> None:
    with tarfile.open(output, "w") as archive:
        for name in names:
            source = files_by_name[name]
            archive.add(source, arcname=name, recursive=False)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_call(base_url: str, token: str, path: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}{path}",
                headers={"X-Worker-Token": token},
                json=payload,
                timeout=timeout,
            )
            if response.ok:
                return response.json()
            if response.status_code < 500:
                raise MigrationError(
                    f"Destination worker returned {response.status_code}: {response.text[:500]}"
                )
            last_error = MigrationError(
                f"Destination worker returned {response.status_code}: {response.text[:500]}"
            )
        except requests.RequestException as exc:
            last_error = exc
        if attempt < 4:
            time.sleep(2 ** attempt)
    raise MigrationError(f"Destination worker unavailable after retries: {last_error}") from last_error


def _rclone_upload(
    chunk: Path,
    config_path: Path,
    remote_path: str,
    *,
    transfers: int = 1,
) -> None:
    result = subprocess.run(
        [
            "rclone", "copyto", str(chunk), f"target:{remote_path}",
            "--config", str(config_path),
            "--transfers", str(transfers),
            "--checkers", "4",
            "--retries", "5",
            "--low-level-retries", "10",
        ],
        capture_output=True,
        text=True,
        timeout=7 * 24 * 3600,
    )
    if result.returncode:
        raise MigrationError(result.stderr.strip() or result.stdout.strip() or "rclone migration upload failed")


def migrate_to_destination(
    *,
    target_url: str,
    target_token: str,
    target_ssh_host: str,
    target_ssh_port: int,
    target_ssh_user: str,
    ssh_private_key: str,
    target_path: str = "/workspace",
    chunk_size_mb: int = 1024,
    include_movies: bool = True,
    include_clips: bool = True,
    include_frames: bool = True,
    include_embeddings: bool = True,
    include_metadata: bool = True,
    migration_id: str | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    if not target_ssh_host or not target_url or not ssh_private_key.strip():
        raise MigrationError("Destination URL, SSH host and migration key are required")
    if not 64 <= chunk_size_mb <= 4096:
        raise MigrationError("chunk_size_mb must be between 64 and 4096")
    migration_id = migration_id or f"mig-{time.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    chunk_bytes = chunk_size_mb * 1024 * 1024
    source_free = shutil.disk_usage(LIBRARY_DIR.parent).free
    if source_free < chunk_bytes:
        raise MigrationError(
            f"Source needs at least one {chunk_size_mb} MB temporary chunk, "
            f"but only {source_free} bytes are free"
        )

    running_marker = LIBRARY_DIR.parent / f".migration-source-{migration_id}.json"
    temp_root = Path(tempfile.mkdtemp(prefix=f"migration-{migration_id}-", dir=str(LIBRARY_DIR.parent)))
    key_path = temp_root / "migration_key"
    config_path = temp_root / "rclone.conf"
    request_state = {
        "target_url": target_url,
        "target_token": target_token,
        "target_ssh_host": target_ssh_host,
        "target_ssh_port": target_ssh_port,
        "target_ssh_user": target_ssh_user,
        "ssh_private_key": ssh_private_key,
        "target_path": target_path,
        "chunk_size_mb": chunk_size_mb,
        "include_movies": include_movies,
        "include_clips": include_clips,
        "include_frames": include_frames,
        "include_embeddings": include_embeddings,
        "include_metadata": include_metadata,
        "migration_id": migration_id,
    }
    running_marker.write_text(json.dumps({"request": request_state}, indent=2), encoding="utf-8")
    running_marker.chmod(0o600)
    succeeded = False
    try:
        key_path.write_text(ssh_private_key, encoding="utf-8")
        key_path.chmod(0o600)
        config_path.write_text(
            "[target]\n"
            "type = sftp\n"
            f"host = {target_ssh_host}\n"
            f"port = {int(target_ssh_port)}\n"
            f"user = {target_ssh_user}\n"
            f"key_file = {key_path}\n"
            "use_insecure_cipher = false\n",
            encoding="utf-8",
        )
        config_path.chmod(0o600)

        if progress:
            progress("Pausing source jobs", 0.01)
        from .processor import pause_processing, running_movie_ids
        for movie_id in running_movie_ids():
            pause_processing(movie_id)
        deadline = time.monotonic() + 120
        while running_movie_ids() and time.monotonic() < deadline:
            time.sleep(1)
        if running_movie_ids():
            raise MigrationError("Source jobs did not stop within 120 seconds")

        if progress:
            progress("Preparing consistent library snapshot", 0.03)
        db_copy = temp_root / "movie_clips.sqlite3"
        _sqlite_snapshot(db_copy)
        files = _files_for_migration(
            include_movies, include_clips, include_frames,
            include_embeddings, include_metadata, db_copy,
        )
        files_by_name = {name: path for path, name in files}
        chunks = _plan_chunks(files, chunk_bytes)
        total_bytes = sum(item["bytes"] for item in chunks)
        manifest = {
            "format": MIGRATION_FORMAT,
            "migration_id": migration_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunk_size_mb": chunk_size_mb,
            "total_bytes": total_bytes,
            "included": [
                name for name, enabled in (
                    ("movies", include_movies), ("clips", include_clips),
                    ("frames", include_frames), ("embeddings", include_embeddings),
                    ("metadata", include_metadata),
                ) if enabled
            ],
            "chunks": chunks,
        }
        running_marker.write_text(
            json.dumps({"request": request_state, "manifest": manifest}, indent=2),
            encoding="utf-8",
        )
        running_marker.chmod(0o600)
        _target_call(target_url, target_token, "/migrations/prepare", {
            "migration_id": migration_id,
            "manifest": manifest,
            "target_path": target_path,
            "confirmation": f"MIGRATE {migration_id}",
        })

        completed = set()
        state = _target_call(target_url, target_token, f"/migrations/{migration_id}/status", {})
        completed.update(state.get("completed_chunks", []))
        with tempfile.TemporaryDirectory(prefix="migration-chunk-", dir=str(temp_root)) as chunk_dir:
            chunk_dir_path = Path(chunk_dir)
            for index, item in enumerate(chunks):
                name = item["name"]
                if name in completed:
                    continue
                if progress:
                    progress(f"Transferring {name}", 0.05 + 0.9 * index / max(1, len(chunks)))
                output = chunk_dir_path / name
                _make_chunk(files_by_name, item["files"], output)
                digest = _hash_file(output)
                remote_file = f"{target_path.rstrip('/')}/.migration-transfer/{migration_id}/{name}"
                _rclone_upload(output, config_path, remote_file)
                _target_call(target_url, target_token, f"/migrations/{migration_id}/chunk", {
                    "name": name,
                    "remote_path": remote_file,
                    "sha256": digest,
                    "size": output.stat().st_size,
                }, timeout=3600)
                output.unlink(missing_ok=True)

        if progress:
            progress("Validating and activating destination", 0.97)
        result = _target_call(target_url, target_token, f"/migrations/{migration_id}/finalize", {}, timeout=3600)
        # The destination now owns the new library.  Restart it automatically
        # so its in-memory DB/model state cannot point at the old library.
        try:
            _target_call(target_url, target_token, "/restart", {}, timeout=30)
            result["restart_requested"] = True
        except MigrationError as exc:
            # Activation already succeeded; report the restart issue without
            # turning a successful data migration into a false failure.
            result["restart_requested"] = False
            result["restart_error"] = str(exc)
        if progress:
            progress("Migration complete", 1.0)
        succeeded = True
        return result
    finally:
        if succeeded:
            running_marker.unlink(missing_ok=True)
        shutil.rmtree(temp_root, ignore_errors=True)


def prepare_destination(*, migration_id: str, manifest: dict[str, Any], target_path: str, confirmation: str) -> dict[str, Any]:
    if confirmation != f"MIGRATE {migration_id}":
        raise MigrationError("Migration confirmation does not match")
    if manifest.get("format") != MIGRATION_FORMAT or manifest.get("migration_id") != migration_id:
        raise MigrationError("Unsupported migration manifest")
    transport_root = Path(target_path).resolve()
    if transport_root != LIBRARY_DIR.parent.resolve():
        raise MigrationError(
            f"Migration target path must be {LIBRARY_DIR.parent}, not {target_path}"
        )
    total_bytes = int(manifest.get("total_bytes") or 0)
    chunk_size = max(1, int(manifest.get("chunk_size_mb") or 1)) * 1024 * 1024
    usage = shutil.disk_usage(LIBRARY_DIR.parent)
    if usage.free < total_bytes + chunk_size:
        raise MigrationError(
            f"Destination needs {total_bytes + chunk_size} bytes free, only {usage.free} are available"
        )
    staging = LIBRARY_DIR.parent / f".library.migration-{migration_id}"
    staging.mkdir(parents=True, exist_ok=True)
    state_path = LIBRARY_DIR.parent / f".migration-destination-{migration_id}.json"
    state = {
        "migration_id": migration_id,
        "manifest": manifest,
        "staging": str(staging),
        "transfer_root": str(transport_root / ".migration-transfer" / migration_id),
        "completed_chunks": [],
    }
    if state_path.exists():
        try:
            old = json.loads(state_path.read_text(encoding="utf-8"))
            if old.get("manifest", {}).get("chunks") == manifest.get("chunks"):
                state = old
        except (OSError, json.JSONDecodeError):
            pass
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    (LIBRARY_DIR.parent / ".migration-transfer" / migration_id).mkdir(parents=True, exist_ok=True)
    return {"migration_id": migration_id, "staging": str(staging), "completed_chunks": state["completed_chunks"]}


def destination_status(migration_id: str) -> dict[str, Any]:
    path = LIBRARY_DIR.parent / f".migration-destination-{migration_id}.json"
    if not path.exists():
        raise MigrationError("Migration is not prepared")
    state = json.loads(path.read_text(encoding="utf-8"))
    return {
        "migration_id": migration_id,
        "completed_chunks": state.get("completed_chunks", []),
        "total_chunks": len(state.get("manifest", {}).get("chunks", [])),
    }


def receive_chunk(*, migration_id: str, name: str, remote_path: str, sha256: str, size: int) -> dict[str, Any]:
    state_path = LIBRARY_DIR.parent / f".migration-destination-{migration_id}.json"
    if not state_path.exists():
        raise MigrationError("Migration is not prepared")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected = {item["name"]: item for item in state["manifest"].get("chunks", [])}
    if name not in expected:
        raise MigrationError(f"Unexpected migration chunk: {name}")
    if name in state.get("completed_chunks", []):
        return {"name": name, "status": "already-complete"}
    transfer_root = Path(state.get("transfer_root", LIBRARY_DIR.parent / ".migration-transfer" / migration_id)).resolve()
    path = Path(remote_path).resolve()
    if path.parent != transfer_root or path.name != name or not path.is_file():
        raise MigrationError("Invalid migration chunk path")
    if path.stat().st_size != int(size) or _hash_file(path) != sha256:
        raise MigrationError(f"Migration chunk verification failed: {name}")
    staging = Path(state["staging"])
    with tarfile.open(path, "r") as archive:
        _safe_members(archive, staging)
        archive.extractall(staging)
    path.unlink(missing_ok=True)
    state.setdefault("completed_chunks", []).append(name)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return {"name": name, "status": "complete", "completed_chunks": state["completed_chunks"]}


def finalize_destination(migration_id: str) -> dict[str, Any]:
    state_path = LIBRARY_DIR.parent / f".migration-destination-{migration_id}.json"
    if not state_path.exists():
        raise MigrationError("Migration is not prepared")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected = {item["name"] for item in state["manifest"].get("chunks", [])}
    completed = set(state.get("completed_chunks", []))
    if completed != expected:
        raise MigrationError(f"Migration incomplete: {len(completed)}/{len(expected)} chunks received")
    staging = Path(state["staging"])
    restored_db = staging / "movie_clips.sqlite3"
    if not restored_db.is_file():
        raise MigrationError("Migrated SQLite database is missing")
    with sqlite3.connect(restored_db) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise MigrationError(f"Migrated SQLite integrity check failed: {integrity}")
    rollback = LIBRARY_DIR.parent / f"{LIBRARY_DIR.name}.rollback-{migration_id}"
    if rollback.exists():
        raise MigrationError(f"Rollback path already exists: {rollback}")
    if LIBRARY_DIR.exists():
        LIBRARY_DIR.replace(rollback)
    try:
        staging.replace(LIBRARY_DIR)
    except Exception:
        if rollback.exists() and not LIBRARY_DIR.exists():
            rollback.replace(LIBRARY_DIR)
        raise
    transfer_root = Path(state.get("transfer_root", LIBRARY_DIR.parent / ".migration-transfer" / migration_id))
    transfer_root.exists() and shutil.rmtree(transfer_root, ignore_errors=True)
    state_path.unlink(missing_ok=True)
    return {
        "migration_id": migration_id,
        "library": str(LIBRARY_DIR),
        "rollback": str(rollback),
        "integrity": integrity,
        "restart_recommended": True,
    }


def pending_source_migrations() -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for path in sorted(LIBRARY_DIR.parent.glob(".migration-source-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            request = payload.get("request")
            if isinstance(request, dict) and request.get("migration_id"):
                pending.append(request)
        except (OSError, json.JSONDecodeError):
            continue
    return pending
