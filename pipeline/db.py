from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import DB_PATH, ensure_library_dirs
from .filter_query import FilterClause, parse_filter_query
from .profiles import BUILTIN_PROFILES


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
                collection_title TEXT DEFAULT '',
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

            CREATE TABLE IF NOT EXISTS embedding_profiles (
                profile_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                input_size INTEGER NOT NULL,
                default_embeddings_per_clip INTEGER NOT NULL,
                frames_per_embedding INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS clip_embeddings (
                clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
                profile_id TEXT NOT NULL REFERENCES embedding_profiles(profile_id) ON DELETE CASCADE,
                artifact_path TEXT NOT NULL DEFAULT '',
                frame_count INTEGER NOT NULL DEFAULT 0,
                dimension INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'missing',
                error TEXT NOT NULL DEFAULT '',
                source_checksum TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (clip_id, profile_id)
            );

            CREATE INDEX IF NOT EXISTS idx_clips_movie ON clips(movie_id);
            CREATE INDEX IF NOT EXISTS idx_clips_duration ON clips(duration);
            CREATE INDEX IF NOT EXISTS idx_clips_camera ON clips(camera_motion_type);
            CREATE INDEX IF NOT EXISTS idx_clips_people ON clips(people_count);
            CREATE INDEX IF NOT EXISTS idx_clips_shot_size ON clips(shot_size);
            CREATE INDEX IF NOT EXISTS idx_clip_embeddings_profile ON clip_embeddings(profile_id, status);
            """
        )
        _migrate(conn)
        _sync_builtin_profiles(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced by the GPU worker to databases created earlier."""
    added = {
        "movies": {
            "source_url": "TEXT DEFAULT ''",
            "collection_title": "TEXT DEFAULT ''",
            "encoder": "TEXT DEFAULT ''",
            "device": "TEXT DEFAULT ''",
            "active_embedding_profile": "TEXT DEFAULT ''",
            "embeddings_per_clip": "INTEGER DEFAULT 0",
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


def _sync_builtin_profiles(conn: sqlite3.Connection) -> None:
    now = utc_now()
    for profile in BUILTIN_PROFILES:
        conn.execute(
            """
            INSERT INTO embedding_profiles
                (profile_id, label, model_name, model_type, input_size,
                 default_embeddings_per_clip, frames_per_embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id) DO UPDATE SET
                label = excluded.label,
                model_name = excluded.model_name,
                model_type = excluded.model_type,
                input_size = excluded.input_size,
                default_embeddings_per_clip = excluded.default_embeddings_per_clip,
                frames_per_embedding = excluded.frames_per_embedding,
                updated_at = excluded.updated_at
            """,
            (
                profile.id,
                profile.label,
                profile.model_name,
                profile.model_type,
                profile.input_size,
                profile.default_embeddings_per_clip,
                profile.frames_per_embedding,
                now,
                now,
            ),
        )


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
    collection_title: str = "",
) -> int:
    now = utc_now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO movies
                (original_name, filename, path, checksum, duration, fps, width, height, collection_title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (original_name, filename, str(path), checksum, duration, fps, width, height, collection_title, now, now),
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


def find_movie_by_checksum(checksum: str, *, exclude_id: int | None = None) -> dict[str, Any] | None:
    if not checksum:
        return None
    sql = "SELECT * FROM movies WHERE checksum = ?"
    values: list[Any] = [checksum]
    if exclude_id is not None:
        sql += " AND id != ?"
        values.append(exclude_id)
    sql += " ORDER BY id DESC LIMIT 1"
    with connect() as conn:
        return row_to_dict(conn.execute(sql, values).fetchone())


def find_movie_by_source_url(source_url: str) -> dict[str, Any] | None:
    if not source_url:
        return None
    with connect() as conn:
        return row_to_dict(
            conn.execute(
                "SELECT * FROM movies WHERE source_url = ? ORDER BY id DESC LIMIT 1",
                (source_url,),
            ).fetchone()
        )


def list_movies() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM movies ORDER BY created_at DESC").fetchall()
    return [row_to_dict(row) for row in rows if row is not None]


def list_collection_titles() -> list[str]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT collection_title
            FROM movies
            WHERE collection_title IS NOT NULL AND collection_title != ''
            ORDER BY collection_title COLLATE NOCASE
            """
        ).fetchall()
    return [str(row["collection_title"]) for row in rows]


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


def delete_clips_for_movie(movie_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM clips WHERE movie_id = ?", (movie_id,))


def delete_movie(movie_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM movies WHERE id = ?", (movie_id,))


def list_embedding_profiles(movie_id: int | None = None) -> list[dict[str, Any]]:
    values: list[Any] = []
    movie_clause = ""
    if movie_id is not None:
        movie_clause = " AND c.movie_id = ?"
        values.append(movie_id)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT p.*,
                   COUNT(CASE WHEN ce.status = 'complete' AND c.id IS NOT NULL THEN 1 END) AS complete_count,
                   COUNT(CASE WHEN ce.status = 'failed' AND c.id IS NOT NULL THEN 1 END) AS failed_count
            FROM embedding_profiles p
            LEFT JOIN clip_embeddings ce ON ce.profile_id = p.profile_id
            LEFT JOIN clips c ON c.id = ce.clip_id{movie_clause}
            GROUP BY p.profile_id
            ORDER BY p.label COLLATE NOCASE
            """,
            values,
        ).fetchall()
        if movie_id is None:
            target_count = int(conn.execute("SELECT COUNT(*) FROM clips WHERE status != 'too_short'").fetchone()[0])
        else:
            target_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM clips WHERE movie_id = ? AND status != 'too_short'",
                    (movie_id,),
                ).fetchone()[0]
            )
    out = [dict(row) for row in rows]
    for item in out:
        item["target_count"] = target_count
        item["missing_count"] = max(0, target_count - int(item["complete_count"] or 0))
    return out


def get_clip_embedding(clip_id: int, profile_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM clip_embeddings WHERE clip_id = ? AND profile_id = ?",
            (clip_id, profile_id),
        ).fetchone()
    return dict(row) if row else None


def list_clip_embeddings(profile_id: str, movie_id: int | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT ce.*, c.movie_id, c.clip_index, c.clip_path, c.start_time, c.end_time,
               c.duration, m.collection_title, m.original_name, m.checksum AS movie_checksum
        FROM clip_embeddings ce
        JOIN clips c ON c.id = ce.clip_id
        JOIN movies m ON m.id = c.movie_id
        WHERE ce.profile_id = ?
    """
    values: list[Any] = [profile_id]
    if movie_id is not None:
        sql += " AND c.movie_id = ?"
        values.append(movie_id)
    sql += " ORDER BY c.movie_id, c.clip_index"
    with connect() as conn:
        rows = conn.execute(sql, values).fetchall()
    return [dict(row) for row in rows]


def upsert_clip_embedding(
    clip_id: int,
    profile_id: str,
    *,
    artifact_path: str = "",
    frame_count: int = 0,
    dimension: int = 0,
    status: str,
    error: str = "",
    source_checksum: str = "",
) -> None:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO clip_embeddings
                (clip_id, profile_id, artifact_path, frame_count, dimension, status,
                 error, source_checksum, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id, profile_id) DO UPDATE SET
                artifact_path = excluded.artifact_path,
                frame_count = excluded.frame_count,
                dimension = excluded.dimension,
                status = excluded.status,
                error = excluded.error,
                source_checksum = excluded.source_checksum,
                updated_at = excluded.updated_at
            """,
            (
                clip_id,
                profile_id,
                artifact_path,
                frame_count,
                dimension,
                status,
                error,
                source_checksum,
                now,
                now,
            ),
        )


def _clip_where(filters: dict[str, Any] | None = None) -> tuple[list[str], list[Any]]:
    filters = filters or {}
    where: list[str] = []
    values: list[Any] = []

    if filters.get("movie_id"):
        where.append("clips.movie_id = ?")
        values.append(filters["movie_id"])
    if filters.get("collection_title"):
        where.append("movies.collection_title = ?")
        values.append(filters["collection_title"])
    if filters.get("min_duration") is not None:
        where.append("clips.duration >= ?")
        values.append(float(filters["min_duration"]))
    if filters.get("max_duration") is not None:
        where.append("clips.duration <= ?")
        values.append(float(filters["max_duration"]))
    if filters.get("camera_motion_type"):
        where.append("clips.camera_motion_type = ?")
        values.append(filters["camera_motion_type"])
    if filters.get("animation_motion_bucket"):
        where.append("clips.animation_motion_bucket = ?")
        values.append(filters["animation_motion_bucket"])
    if filters.get("people_count"):
        where.append("clips.people_count = ?")
        values.append(filters["people_count"])
    if filters.get("shot_size"):
        where.append("clips.shot_size = ?")
        values.append(filters["shot_size"])
    if filters.get("status"):
        where.append("clips.status = ?")
        values.append(filters["status"])
    if filters.get("has_file"):
        where.append("clips.clip_path IS NOT NULL AND clips.clip_path != ''")
    for clause in parse_filter_query((filters.get("filter_query") or "").strip()):
        clause_sql, clause_values = _advanced_clause_sql(clause)
        where.append(clause_sql)
        values.extend(clause_values)
    text = (filters.get("text") or "").strip().lower()
    if text:
        text_like = f"%{text}%"
        where.append(
            """
            (
                LOWER(clips.description) LIKE ?
                OR LOWER(clips.user_notes) LIKE ?
                OR LOWER(clips.tags) LIKE ?
                OR LOWER(clips.moods) LIKE ?
                OR LOWER(clips.settings) LIKE ?
            )
            """
        )
        values.extend([text_like] * 5)
    mood = (filters.get("mood") or "").strip().lower()
    if mood:
        where.append("LOWER(clips.moods) LIKE ?")
        values.append(f'%"{mood}"%')
    tag = (filters.get("tag") or "").strip().lower()
    if tag:
        where.append("LOWER(clips.tags) LIKE ?")
        values.append(f'%"{tag}"%')

    return where, values


def _advanced_clause_sql(clause: FilterClause) -> tuple[str, list[Any]]:
    values = list(clause.values)
    if clause.field == "title":
        placeholders = ", ".join("?" for _ in values)
        keyword = "NOT IN" if clause.operator == "!=" else "IN"
        return f"LOWER(movies.collection_title) {keyword} ({placeholders})", [value.lower() for value in values]

    if clause.field == "people":
        expression = "CASE clips.people_count WHEN 'none' THEN 0 WHEN 'one' THEN 1 WHEN 'two' THEN 2 WHEN 'group' THEN 3 ELSE -1 END"
        if len(values) > 1:
            placeholders = ", ".join("?" for _ in values)
            return f"{expression} IN ({placeholders})", [int(float(value)) for value in values]
        return f"{expression} {clause.operator} ?", [int(float(values[0]))]

    if clause.field in {"minsec", "maxsec", "duration"}:
        operator = clause.operator
        if clause.field == "minsec" and operator == "=":
            operator = ">="
        elif clause.field == "maxsec" and operator == "=":
            operator = "<="
        return f"clips.duration {operator} ?", [float(values[0])]

    if clause.field in {"shot", "camera", "motion"}:
        column = {
            "shot": "clips.shot_size",
            "camera": "clips.camera_motion_type",
            "motion": "clips.animation_motion_bucket",
        }[clause.field]
        placeholders = ", ".join("?" for _ in values)
        keyword = "NOT IN" if clause.operator == "!=" else "IN"
        return f"LOWER({column}) {keyword} ({placeholders})", [value.lower() for value in values]

    if clause.field in {"mood", "tag"}:
        column = "clips.moods" if clause.field == "mood" else "clips.tags"
        comparisons = []
        sql_values: list[Any] = []
        for value in values:
            comparisons.append(f"LOWER({column}) {'NOT LIKE' if clause.operator == '!=' else 'LIKE'} ?")
            sql_values.append(f'%"{value.lower()}"%')
        joiner = " AND " if clause.operator == "!=" else " OR "
        return f"({joiner.join(comparisons)})", sql_values

    if clause.field == "files":
        enabled = values[0].lower() in {"true", "yes", "1"}
        if clause.operator == "!=":
            enabled = not enabled
        if enabled:
            return "(clips.clip_path IS NOT NULL AND clips.clip_path != '')", []
        return "(clips.clip_path IS NULL OR clips.clip_path = '')", []

    raise ValueError(f"Unsupported advanced filter field: {clause.field}")


def count_clips(filters: dict[str, Any] | None = None) -> int:
    where, values = _clip_where(filters)
    sql = "SELECT COUNT(*) AS count FROM clips LEFT JOIN movies ON movies.id = clips.movie_id"
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

    sql = """
        SELECT clips.*, movies.collection_title AS collection_title,
               movies.original_name AS movie_original_name
        FROM clips
        LEFT JOIN movies ON movies.id = clips.movie_id
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY clips.movie_id DESC, clips.start_time ASC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        values.extend([max(0, int(limit)), max(0, int(offset))])

    with connect() as conn:
        rows = conn.execute(sql, values).fetchall()

    return [row_to_dict(row) for row in rows if row is not None]
