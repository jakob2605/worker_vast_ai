from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import db
from .config import DB_PATH, DOWNLOAD_LINKS_PATH, LIBRARY_DIR

Progress = Callable[[str, float], None]


def remote_root() -> str:
    return os.getenv("RCLONE_REMOTE", "gdrive:VastAIProgram").rstrip("/")


def rclone_ready() -> dict[str, Any]:
    executable = shutil.which("rclone")
    if not executable:
        return {"ready": False, "error": "rclone is not installed", "remote": remote_root()}
    result = subprocess.run([executable, "listremotes"], capture_output=True, text=True, timeout=30)
    configured = {line.strip().rstrip(":") for line in result.stdout.splitlines() if line.strip()}
    remote_name = remote_root().split(":", 1)[0]
    return {
        "ready": result.returncode == 0 and remote_name in configured,
        "remote": remote_root(),
        "configured_remotes": sorted(configured),
        "error": result.stderr.strip() if result.returncode else "",
    }


def _run(args: list[str], timeout: int = 86400) -> None:
    result = subprocess.run(["rclone", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "rclone failed")


def list_snapshots() -> list[dict[str, Any]]:
    ready = rclone_ready()
    if not ready["ready"]:
        return []
    result = subprocess.run(
        ["rclone", "lsjson", f"{remote_root()}/snapshots"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        return []
    try:
        rows = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return []
    snapshots: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("Name")
        if not name:
            continue
        is_archive = name.endswith(".zip") and not row.get("IsDir")
        is_folder = bool(row.get("IsDir"))
        if not is_archive and not is_folder:
            continue
        snapshot_id = name.removesuffix(".zip")
        if snapshot_id in snapshots and snapshots[snapshot_id].get("archive") and is_folder:
            continue
        snapshots[snapshot_id] = {
            "id": snapshot_id,
            "modified": row.get("ModTime"),
            "archive": is_archive,
        }
    return sorted(snapshots.values(), key=lambda item: item["id"], reverse=True)


def _add_tree_to_zip(zf: zipfile.ZipFile, source: Path, prefix: str) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        if path.is_file():
            zf.write(path, Path(prefix) / path.relative_to(source))


def _extract_zip_safely(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe path in snapshot archive: {member.filename}")
        zf.extractall(destination)


def create_snapshot(
    *,
    include_movies: bool = False,
    include_frames: bool = False,
    zip_archive: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    ready = rclone_ready()
    if not ready["ready"]:
        raise RuntimeError(ready["error"] or f"rclone remote {remote_root()} is not configured")
    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = LIBRARY_DIR.parent / ".backup-staging" / snapshot_id
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    def update(message: str, value: float) -> None:
        if progress:
            progress(message, value)

    update("SQLite snapshot", 0.05)
    target_db = staging / DB_PATH.name
    with sqlite3.connect(DB_PATH) as source, sqlite3.connect(target_db) as target:
        source.backup(target)
    if DOWNLOAD_LINKS_PATH.exists():
        shutil.copy2(DOWNLOAD_LINKS_PATH, staging / DOWNLOAD_LINKS_PATH.name)

    included = ["clips", "metadata", "embeddings"]
    if include_frames:
        included.append("frames")
    if include_movies:
        included.append("movies")
    manifest = {
        "format": "vastai-library-snapshot-v1",
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "included": included,
        "include_movies": include_movies,
        "include_frames": include_frames,
        "zip_archive": zip_archive,
        "profiles": db.list_embedding_profiles(),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not zip_archive:
        destination = f"{remote_root()}/snapshots/{snapshot_id}"
        _run(["copy", str(staging), destination, "--checksum"])

        for index, folder in enumerate(included, start=1):
            source = LIBRARY_DIR / folder
            if source.exists():
                update(f"Google Drive: {folder}", 0.1 + 0.85 * index / len(included))
                _run([
                    "copy",
                    str(source),
                    f"{destination}/{folder}",
                    "--checksum",
                    "--transfers",
                    "8",
                    "--checkers",
                    "16",
                ])
        update("Snapshot complete", 1.0)
        shutil.rmtree(staging, ignore_errors=True)
        return {"snapshot_id": snapshot_id, "remote": destination, "included": included, "archive": False}

    archive = staging.parent / f"{snapshot_id}.zip"
    archive.unlink(missing_ok=True)
    try:
        update("Snapshot wird gezippt", 0.1)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for file in staging.iterdir():
                if file.is_file():
                    zf.write(file, file.name)
            for index, folder in enumerate(included, start=1):
                update(f"Zip: {folder}", 0.1 + 0.75 * index / len(included))
                _add_tree_to_zip(zf, LIBRARY_DIR / folder, folder)

        destination = f"{remote_root()}/snapshots/{snapshot_id}.zip"
        update("Google Drive: Snapshot-Archiv", 0.9)
        _run(["copyto", str(archive), destination, "--checksum"])
        update("Snapshot complete", 1.0)
        return {"snapshot_id": snapshot_id, "remote": destination, "included": included, "archive": True}
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)


def restore_snapshot(snapshot_id: str, *, progress: Progress | None = None) -> dict[str, Any]:
    snapshots = list_snapshots()
    if snapshot_id == "latest":
        if not snapshots:
            raise RuntimeError("No Google Drive snapshots are available")
        snapshot_id = snapshots[0]["id"]
    if not snapshot_id or any(char not in "0123456789TZ" for char in snapshot_id):
        raise ValueError("Invalid snapshot id")
    ready = rclone_ready()
    if not ready["ready"]:
        raise RuntimeError(ready["error"] or f"rclone remote {remote_root()} is not configured")

    snapshot_info = next((item for item in snapshots if item["id"] == snapshot_id), None)
    if snapshot_info is None:
        raise RuntimeError(f"Google Drive snapshot not found: {snapshot_id}")
    is_archive = bool(snapshot_info.get("archive"))

    staging = LIBRARY_DIR.parent / f".{LIBRARY_DIR.name}.restore-{snapshot_id}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Google Drive snapshot wird geladen", 0.1)
    archive = staging.parent / f".{LIBRARY_DIR.name}.restore-{snapshot_id}.zip"
    archive.unlink(missing_ok=True)
    try:
        if is_archive:
            if progress:
                progress("Google Drive: Snapshot-Archiv wird geladen", 0.1)
            _run(["copyto", f"{remote_root()}/snapshots/{snapshot_id}.zip", str(archive), "--checksum"])
            if progress:
                progress("Snapshot-Archiv wird entpackt", 0.5)
            _extract_zip_safely(archive, staging)
        else:
            if progress:
                progress("Google Drive: Snapshot-Ordner wird kopiert", 0.1)
            _run([
                "copy",
                f"{remote_root()}/snapshots/{snapshot_id}",
                str(staging),
                "--checksum",
                "--transfers",
                "8",
                "--checkers",
                "16",
            ])
    finally:
        archive.unlink(missing_ok=True)
    manifest_path = staging / "manifest.json"
    restored_db = staging / DB_PATH.name
    if not manifest_path.exists() or not restored_db.exists():
        raise RuntimeError("Snapshot is incomplete: manifest or SQLite database is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "vastai-library-snapshot-v1":
        raise RuntimeError("Unsupported snapshot format")
    with sqlite3.connect(restored_db) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"Restored SQLite database failed integrity check: {integrity}")

    rollback = LIBRARY_DIR.parent / f"{LIBRARY_DIR.name}.rollback-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if progress:
        progress("Bibliothek wird atomar umgeschaltet", 0.9)
    if LIBRARY_DIR.exists():
        LIBRARY_DIR.replace(rollback)
    try:
        staging.replace(LIBRARY_DIR)
    except Exception:
        if rollback.exists() and not LIBRARY_DIR.exists():
            rollback.replace(LIBRARY_DIR)
        raise
    if progress:
        progress("Restore complete", 1.0)
    return {
        "snapshot_id": snapshot_id,
        "library": str(LIBRARY_DIR),
        "rollback": str(rollback),
        "integrity": integrity,
        "restart_recommended": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("--include-movies", action="store_true")
    backup.add_argument("--include-frames", action="store_true")
    backup.add_argument("--zip", action="store_true", dest="zip_archive")
    restore = sub.add_parser("restore")
    restore.add_argument("snapshot")
    args = parser.parse_args()
    if args.command == "backup":
        result = create_snapshot(
            include_movies=args.include_movies,
            include_frames=args.include_frames,
            zip_archive=args.zip_archive,
        )
    else:
        result = restore_snapshot(args.snapshot)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
