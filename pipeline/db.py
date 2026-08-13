from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import DB_PATH, ensure_library_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterable[sqlite3.Connection]:
    ensure_library_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_name TEXT NOT NULL,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                duration REAL DEFAULT 0,
                fps REAL DEFAULT 0,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'imported',
                progress_stage TEXT NOT NULL DEFAULT 'imported',
                progress_detail TEXT NOT NULL DEFAULT '',
                paused INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                detector TEXT,
                source_url TEXT DEFAULT '',
                encoder TEXT DEFAULT '',
                device TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
                clip_index INTEGER NOT NULL,
                clip_path TEXT,
                metadata_path TEXT,
                embedding_path TEXT,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'detected',
                camera_motion_type TEXT DEFAULT 'unknown',
                camera_motion_score REAL DEFAULT 0,
                camera_translation_px_sec REAL DEFAULT 0,
                camera_rotation_deg_sec REAL DEFAULT 0,
                camera_zoom_delta REAL DEFAULT 0,
                camera_confidence REAL DEFAULT 0,
                animation_motion_score REAL DEFAULT 0,
                animation_motion_bucket TEXT DEFAULT 'unknown',
                people_count TEXT DEFAULT 'unknown',
                shot_size TEXT DEFAULT 'unknown',
                moods TEXT NOT NULL DEFAULT '[]',
                settings TEXT NOT NULL DEFAULT '[]',
                quality_flags TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                user_notes TEXT NOT NULL DEFAULT '',
                downloaded_at TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(movie_id, clip_index)
            );

            CREATE INDEX IF NOT EXISTS idx_clips_movie ON clips(movie_id);
            CREATE INDEX IF NOT EXISTS idx_clips_duration ON clips(duration);
            CREATE INDEX IF NOT EXISTS idx_clips_camera ON clips(camera_motion_type);
            CREATE INDEX IF NOT EXISTS idx_clips_people ON clips(people_count);
            CREATE INDEX IF NOT EXISTS idx_clips_shot_size ON clips(shot_size);
            """
        )
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced by the GPU worker to databases created earlier."""
    added = {
        "movies": {
            "source_url": "TEXT DEFAULT ''",
            "encoder": "TEXT DEFAULT ''",
            "device": "TEXT DEFAULT ''",
        },
        "clips": {
            "downloaded_at": "TEXT DEFAULT ''",
        },
    }
    for table, columns in added.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in ["moods", "settings", "quality_flags", "tags"]:
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                data[key] = []
    return data


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def create_movie(
    *,
    original_name: str,
    filename: str,
    path: Path,
    checksum: str,
    duration: float,
    fps: float,
    width: int,
    height: int,
) -> int:
    now = utc_now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO movies
                (original_name, filename, path, checksum, duration, fps, width, height, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (original_name, filename, str(path), checksum, duration, fps, width, height, now, now),
        )
        return int(cur.lastrowid)


def update_movie(movie_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [movie_id]
    with connect() as conn:
        conn.execute(f"UPDATE movies SET {assignments} WHERE id = ?", values)


def get_movie(movie_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        return row_to_dict(conn.execute("SELECT * FROM movies WHERE id = ?", (movie_id,)).fetchone())


def list_movies() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM movies ORDER BY created_at DESC").fetchall()
    return [row_to_dict(row) for row in rows if row is not None]


def upsert_clip(movie_id: int, clip_index: int, **fields: Any) -> int:
    now = utc_now()
    fields.setdefault("created_at", now)
    fields["updated_at"] = now
    for key in ["moods", "settings", "quality_flags", "tags"]:
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json_text(fields[key])

    columns = ["movie_id", "clip_index", *fields.keys()]
    values = [movie_id, clip_index, *fields.values()]
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{key} = excluded.{key}" for key in fields if key != "created_at")

    with connect() as conn:
        conn.execute(
            f"""
            INSERT INTO clips ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(movie_id, clip_index) DO UPDATE SET {updates}
            """,
            values,
        )
        row = conn.execute(
            "SELECT id FROM clips WHERE movie_id = ? AND clip_index = ?",
            (movie_id, clip_index),
        ).fetchone()
        return int(row["id"])


def update_clip(clip_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utc_now()
    for key in ["moods", "settings", "quality_flags", "tags"]:
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json_text(fields[key])
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [clip_id]
    with connect() as conn:
        conn.execute(f"UPDATE clips SET {assignments} WHERE id = ?", values)


def get_clip(clip_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        return row_to_dict(conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone())


def delete_clips(clip_ids: list[int]) -> list[dict[str, Any]]:
    if not clip_ids:
        return []
    placeholders = ", ".join("?" for _ in clip_ids)
    with connect() as conn:
        rows = conn.execute(f"SELECT * FROM clips WHERE id IN ({placeholders})", clip_ids).fetchall()
        clips = [row_to_dict(row) for row in rows if row is not None]
        conn.execute(f"DELETE FROM clips WHERE id IN ({placeholders})", clip_ids)
    return clips


def _clip_where(filters: dict[str, Any] | None = None) -> tuple[list[str], list[Any]]:
    filters = filters or {}
    where: list[str] = []
    values: list[Any] = []

    if filters.get("movie_id"):
        where.append("movie_id = ?")
        values.append(filters["movie_id"])
    if filters.get("min_duration") is not None:
        where.append("duration >= ?")
        values.append(float(filters["min_duration"]))
    if filters.get("max_duration") is not None:
        where.append("duration <= ?")
        values.append(float(filters["max_duration"]))
    if filters.get("camera_motion_type"):
        where.append("camera_motion_type = ?")
        values.append(filters["camera_motion_type"])
    if filters.get("animation_motion_bucket"):
        where.append("animation_motion_bucket = ?")
        values.append(filters["animation_motion_bucket"])
    if filters.get("people_count"):
        where.append("people_count = ?")
        values.append(filters["people_count"])
    if filters.get("shot_size"):
        where.append("shot_size = ?")
        values.append(filters["shot_size"])
    if filters.get("status"):
        where.append("status = ?")
        values.append(filters["status"])
    if filters.get("has_file"):
        where.append("clip_path IS NOT NULL AND clip_path != ''")
    text = (filters.get("text") or "").strip().lower()
    if text:
        text_like = f"%{text}%"
        where.append(
            """
            (
                LOWER(description) LIKE ?
                OR LOWER(user_notes) LIKE ?
                OR LOWER(tags) LIKE ?
                OR LOWER(moods) LIKE ?
                OR LOWER(settings) LIKE ?
            )
            """
        )
        values.extend([text_like] * 5)
    mood = (filters.get("mood") or "").strip().lower()
    if mood:
        where.append("LOWER(moods) LIKE ?")
        values.append(f'%"{mood}"%')
    tag = (filters.get("tag") or "").strip().lower()
    if tag:
        where.append("LOWER(tags) LIKE ?")
        values.append(f'%"{tag}"%')

    return where, values


def count_clips(filters: dict[str, Any] | None = None) -> int:
    where, values = _clip_where(filters)
    sql = "SELECT COUNT(*) AS count FROM clips"
    if where:
        sql += " WHERE " + " AND ".join(where)

    with connect() as conn:
        row = conn.execute(sql, values).fetchone()
    return int(row["count"] if row else 0)


def list_clips(
    filters: dict[str, Any] | None = None,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    where, values = _clip_where(filters)

    sql = "SELECT * FROM clips"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY movie_id DESC, start_time ASC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        values.extend([max(0, int(limit)), max(0, int(offset))])

    with connect() as conn:
        rows = conn.execute(sql, values).fetchall()

    return [row_to_dict(row) for row in rows if row is not None]
