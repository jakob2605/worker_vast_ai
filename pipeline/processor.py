from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import uuid
from urllib.parse import unquote, urlparse

from . import db
from .config import CLIPS_DIR, DOWNLOAD_LINKS_PATH, METADATA_DIR, MOVIES_DIR, SETTINGS, ensure_library_dirs
from .motion import analyze_motion
from .semantics import SemanticAnalyzer
from .shot_detection import Shot, detect_shots
from .video_tools import ToolMissingError, download_movie, export_clip, ffprobe, file_sha256


_job_lock = threading.Lock()
_running_jobs: dict[int, threading.Thread] = {}
_download_link_lock = threading.Lock()


def ingest_url(url: str, original_name: str | None = None, collection_title: str = "") -> int:
    """
    Download a movie onto this box and register it, replacing the browser-upload
    path from the local app. Returns the new movie id.
    """
    ensure_library_dirs()
    parsed_name = original_name or unquote(Path(urlparse(url).path).name) or "movie.mp4"
    suffix = Path(parsed_name).suffix.lower() or ".mp4"
    filename = f"{uuid.uuid4().hex}{suffix}"
    target = MOVIES_DIR / filename

    movie_id = db.create_movie(
        original_name=parsed_name,
        filename=filename,
        path=target,
        checksum="",
        duration=0.0,
        fps=0.0,
        width=0,
        height=0,
        collection_title=collection_title,
    )
    record_source_link(movie_id, parsed_name, collection_title, url, target, source_type="url")
    db.update_movie(
        movie_id, status="downloading", progress_stage="downloading",
        progress_detail=f"Fetching {parsed_name}", source_url=url,
    )

    def on_progress(done: int, total: int) -> None:
        pct = (done / total * 100.0) if total else 0.0
        db.update_movie(
            movie_id,
            progress_detail=f"Downloaded {done / 1048576:.0f} MB"
            + (f" of {total / 1048576:.0f} MB ({pct:.1f}%)" if total else ""),
        )

    try:
        download_movie(url, target, progress=on_progress)
        info = ffprobe(target)
        db.update_movie(
            movie_id,
            checksum=file_sha256(target),
            duration=float(info["duration"]),
            fps=float(info["fps"]),
            width=int(info["width"]),
            height=int(info["height"]),
            status="imported",
            progress_stage="imported",
            progress_detail="Ready to process",
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        db.update_movie(movie_id, status="error", progress_stage="error", error=f"Download failed: {exc}")
        raise
    return movie_id


def record_source_link(
    movie_id: int,
    original_name: str,
    collection_title: str,
    source_url: str,
    target: Path,
    *,
    source_type: str,
) -> None:
    entry = {
        "movie_id": movie_id,
        "original_name": original_name,
        "collection_title": collection_title,
        "source_url": source_url,
        "source_type": source_type,
        "target": str(target),
        "created_at": db.utc_now(),
    }
    with _download_link_lock:
        DOWNLOAD_LINKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DOWNLOAD_LINKS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def start_processing(movie_id: int) -> bool:
    with _job_lock:
        thread = _running_jobs.get(movie_id)
        if thread and thread.is_alive():
            return False
        db.update_movie(movie_id, paused=0, status="queued", progress_stage="queued", error=None)
        thread = threading.Thread(target=process_movie, args=(movie_id,), daemon=True)
        _running_jobs[movie_id] = thread
        thread.start()
        return True


def pause_processing(movie_id: int) -> None:
    db.update_movie(movie_id, paused=1, status="paused", progress_stage="paused")


def process_movie(movie_id: int) -> None:
    movie = db.get_movie(movie_id)
    if not movie:
        return

    try:
        source = Path(movie["path"])
        db.update_movie(movie_id, status="processing", progress_stage="detecting_shots", progress_detail="Detecting shot boundaries")
        shots, detector = detect_shots(
            source,
            float(movie["fps"] or 25.0),
            float(movie["duration"] or 0.0),
            SETTINGS.transnet_threshold,
            SETTINGS.merge_tiny_shots_seconds,
        )
        db.update_movie(movie_id, detector=detector)
        _store_shots(movie_id, shots)
        if _is_paused(movie_id):
            return

        db.update_movie(movie_id, progress_stage="exporting_clips", progress_detail="Exporting MP4 clips")
        _export_missing_clips(movie_id, source)
        if _is_paused(movie_id):
            return

        db.update_movie(movie_id, progress_stage="motion_analysis", progress_detail="Analyzing camera and animation movement")
        _analyze_missing_motion(movie_id, source)
        if _is_paused(movie_id):
            return

        db.update_movie(movie_id, progress_stage="semantic_indexing", progress_detail="Creating semantic labels and embeddings")
        _analyze_missing_semantics(movie_id, source)
        if _is_paused(movie_id):
            return

        db.update_movie(movie_id, progress_stage="metadata_export", progress_detail="Writing metadata sidecars")
        _write_all_metadata(movie_id)
        db.update_movie(movie_id, status="complete", progress_stage="complete", progress_detail="Complete", error=None)
    except ToolMissingError as exc:
        db.update_movie(movie_id, status="error", progress_stage="error", error=str(exc))
    except Exception as exc:
        db.update_movie(movie_id, status="error", progress_stage="error", error=repr(exc))
    finally:
        with _job_lock:
            _running_jobs.pop(movie_id, None)


def _store_shots(movie_id: int, shots: list[Shot]) -> None:
    for index, shot in enumerate(shots, start=1):
        if shot.duration < SETTINGS.min_clip_seconds:
            status = "too_short"
        else:
            status = "detected"
        db.upsert_clip(
            movie_id,
            index,
            start_frame=shot.start_frame,
            end_frame=shot.end_frame,
            start_time=shot.start_time,
            end_time=shot.end_time,
            duration=shot.duration,
            status=status,
        )


def _export_missing_clips(movie_id: int, source: Path) -> None:
    clips = [clip for clip in db.list_clips({"movie_id": movie_id}) if clip["status"] not in {"too_short"}]
    movie = db.get_movie(movie_id)
    movie_dir = CLIPS_DIR / _movie_folder_name(movie_id, movie["original_name"] if movie else source.stem)
    for clip in clips:
        if _is_paused(movie_id):
            return
        target = movie_dir / f"clip_{int(clip['clip_index']):05d}_{clip['start_time']:.2f}-{clip['end_time']:.2f}.mp4"
        if not target.exists():
            encoder = export_clip(
                source,
                target,
                float(clip["start_time"]),
                float(clip["end_time"]),
                SETTINGS.export_crf,
                SETTINGS.export_preset,
                use_nvenc=SETTINGS.use_nvenc,
                nvenc_preset=SETTINGS.nvenc_preset,
                nvenc_cq=SETTINGS.nvenc_cq,
            )
            db.update_movie(movie_id, encoder=encoder, device=SETTINGS.device)
        db.update_clip(int(clip["id"]), clip_path=str(target), status="exported")


def _analyze_missing_motion(movie_id: int, source: Path) -> None:
    for clip in db.list_clips({"movie_id": movie_id}):
        if _is_paused(movie_id):
            return
        if clip["status"] in {"too_short"}:
            continue
        metrics = analyze_motion(source, float(clip["start_time"]), float(clip["end_time"]), sample_count=6, resize_width=SETTINGS.motion_resize_width)
        db.update_clip(
            int(clip["id"]),
            status="motion_analyzed",
            camera_motion_type=metrics.camera_motion_type,
            camera_motion_score=metrics.camera_motion_score,
            camera_translation_px_sec=metrics.translation_px_sec,
            camera_rotation_deg_sec=metrics.rotation_deg_sec,
            camera_zoom_delta=metrics.zoom_delta,
            camera_confidence=metrics.confidence,
            animation_motion_score=metrics.animation_motion_score,
            animation_motion_bucket=metrics.animation_motion_bucket,
        )


def _analyze_missing_semantics(movie_id: int, source: Path) -> None:
    analyzer = SemanticAnalyzer()
    for clip in db.list_clips({"movie_id": movie_id}):
        if _is_paused(movie_id):
            return
        if clip["status"] in {"too_short"}:
            continue
        result = analyzer.analyze_clip(source, int(clip["id"]), float(clip["start_time"]), float(clip["end_time"]))
        tags = result["tags"]
        if result.get("semantic_model"):
            tags = list(dict.fromkeys([*tags, result["semantic_model"]]))
        db.update_clip(
            int(clip["id"]),
            status="indexed",
            people_count=result["people_count"],
            shot_size=result["shot_size"],
            moods=result["moods"],
            settings=result["settings"],
            quality_flags=result["quality_flags"],
            tags=tags,
            description=result["description"],
            embedding_path=result["embedding_path"],
        )
        _write_metadata(movie_id, int(clip["id"]), extra={"representative_frames": result.get("frame_paths", [])})


def _write_all_metadata(movie_id: int) -> None:
    for clip in db.list_clips({"movie_id": movie_id}):
        _write_metadata(movie_id, int(clip["id"]))


def _write_metadata(movie_id: int, clip_id: int, extra: dict | None = None) -> None:
    movie = db.get_movie(movie_id)
    clip = db.get_clip(clip_id)
    if not movie or not clip:
        return
    target_dir = METADATA_DIR / _movie_folder_name(movie_id, movie["original_name"])
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"clip_{int(clip['clip_index']):05d}.json"
    payload = {
        "movie": {
            "id": movie["id"],
            "original_name": movie["original_name"],
            "checksum": movie["checksum"],
            "duration": movie["duration"],
            "fps": movie["fps"],
            "width": movie["width"],
            "height": movie["height"],
            "detector": movie["detector"],
        },
        "clip": clip,
    }
    if extra:
        payload.update(extra)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    db.update_clip(clip_id, metadata_path=str(target))


def _is_paused(movie_id: int) -> bool:
    movie = db.get_movie(movie_id)
    paused = bool(movie and movie["paused"])
    if paused:
        db.update_movie(movie_id, status="paused", progress_stage="paused", progress_detail="Paused")
    return paused


def movie_folder_name(movie_id: int, original_name: str) -> str:
    stem = Path(original_name).stem.strip() or f"movie_{movie_id:04d}"
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = f"movie_{movie_id:04d}"
    return f"{cleaned}_{movie_id:04d}"


_movie_folder_name = movie_folder_name
