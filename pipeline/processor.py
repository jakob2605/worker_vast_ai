from __future__ import annotations

import json
import re
import shutil
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

import uuid
from urllib.parse import unquote, urlparse

from . import db
from .config import (
    CLIPS_DIR,
    DOWNLOAD_LINKS_PATH,
    EMBEDDING_PROFILES_DIR,
    FRAMES_DIR,
    METADATA_DIR,
    MOVIES_DIR,
    SETTINGS,
    ensure_library_dirs,
)
from .motion import analyze_motion
from .languagebind import LanguageBindAnalyzer
from .profiles import (
    DEFAULT_PROFILE_ID,
    adaptive_embeddings_per_clip,
    get_profile,
    normalize_embeddings_per_clip,
)
from .semantics import SemanticAnalyzer
from .shot_detection import Shot, detect_shots
from .timing import timing_event
from .video_tools import ToolMissingError, download_movie, export_clip, ffprobe


_job_lock = threading.Lock()
_running_jobs: dict[int, threading.Thread] = {}
_download_link_lock = threading.Lock()
_semantic_lock = threading.RLock()
_semantic_queue_condition = threading.Condition()
_semantic_queue: deque[str] = deque()
_active_semantic_ticket = ""


@contextmanager
def _semantic_queue_slot(movie_id: int, profile_label: str) -> Iterable[float]:
    global _active_semantic_ticket
    ticket = uuid.uuid4().hex
    queued_at = time.perf_counter()
    with _semantic_queue_condition:
        _semantic_queue.append(ticket)
        while _active_semantic_ticket or _semantic_queue[0] != ticket:
            position = list(_semantic_queue).index(ticket) + (1 if _active_semantic_ticket else 0)
            try:
                db.update_movie(
                    movie_id,
                    progress_stage="semantic_indexing",
                    progress_detail=f"Queued for GPU slot: {profile_label} ({position} ahead)",
                )
            except Exception:  # noqa: BLE001
                pass
            _semantic_queue_condition.wait(timeout=2.0)
        _semantic_queue.popleft()
        _active_semantic_ticket = ticket
    try:
        yield time.perf_counter() - queued_at
    finally:
        with _semantic_queue_condition:
            if _active_semantic_ticket == ticket:
                _active_semantic_ticket = ""
            _semantic_queue_condition.notify_all()


def _timed_call(movie_id: int, stage: str, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    started = time.perf_counter()
    timing_event("stage_start", movie_id=movie_id, stage=stage)
    try:
        result = operation(*args, **kwargs)
    except BaseException as exc:
        timing_event(
            "stage_end",
            movie_id=movie_id,
            stage=stage,
            status="error",
            elapsed_s=round(time.perf_counter() - started, 4),
            error=type(exc).__name__,
        )
        raise
    timing_event(
        "stage_end",
        movie_id=movie_id,
        stage=stage,
        status="paused" if _is_paused(movie_id) else "ok",
        elapsed_s=round(time.perf_counter() - started, 4),
    )
    return result


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

    ingest_started = time.perf_counter()
    timing_event("ingest_start", movie_id=movie_id, source_type="url")
    try:
        download_started = time.perf_counter()
        download_movie(url, target, progress=on_progress)
        download_s = time.perf_counter() - download_started
        db.update_movie(movie_id, progress_detail="Inspecting video metadata")
        probe_started = time.perf_counter()
        info = ffprobe(target)
        probe_s = time.perf_counter() - probe_started
        # URL imports are already de-duplicated by source URL before download.
        # Avoid reading a large movie a second time just to calculate SHA-256;
        # retain a lightweight, explicit fingerprint for metadata/embedding
        # provenance. Local browser uploads still use a cryptographic checksum
        # in worker.py because they need content-based duplicate detection.
        file_size = target.stat().st_size
        checksum = (
            f"size:{file_size}:duration:{float(info['duration']):.3f}:"
            f"fps:{float(info['fps']):.3f}:width:{int(info['width'])}:height:{int(info['height'])}"
        )
        checksum_s = 0.0
        db.update_movie(
            movie_id,
            checksum=checksum,
            duration=float(info["duration"]),
            fps=float(info["fps"]),
            width=int(info["width"]),
            height=int(info["height"]),
            status="imported",
            progress_stage="imported",
            progress_detail="Ready to process",
            error=None,
        )
        timing_event(
            "ingest_end",
            movie_id=movie_id,
            source_type="url",
            status="ok",
            download_s=round(download_s, 4),
            probe_s=round(probe_s, 4),
            checksum_s=round(checksum_s, 4),
            checksum_method="size+metadata",
            elapsed_s=round(time.perf_counter() - ingest_started, 4),
        )
    except Exception as exc:  # noqa: BLE001
        timing_event(
            "ingest_end",
            movie_id=movie_id,
            source_type="url",
            status="error",
            elapsed_s=round(time.perf_counter() - ingest_started, 4),
            error=type(exc).__name__,
        )
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


def start_semantics_only(
    movie_id: int,
    profile_id: str = DEFAULT_PROFILE_ID,
    embeddings_per_clip: int | None = None,
    overwrite: bool = False,
    sampling_mode: str = "adaptive",
    adaptive_seconds: float = 1.5,
    adaptive_min: int = 3,
    adaptive_max: int = 16,
) -> bool:
    profile = get_profile(profile_id)
    count = normalize_embeddings_per_clip(embeddings_per_clip, profile)
    mode = "fixed" if sampling_mode == "fixed" else "adaptive"
    if mode == "adaptive":
        adaptive_embeddings_per_clip(
            0,
            seconds_per_vector=adaptive_seconds,
            minimum=adaptive_min,
            maximum=adaptive_max,
        )
    with _job_lock:
        thread = _running_jobs.get(movie_id)
        if thread and thread.is_alive():
            return False
        db.update_movie(
            movie_id,
            paused=0,
            status="processing",
            progress_stage="semantic_indexing",
            progress_detail=f"Waiting to generate {profile.label}",
            active_embedding_profile=profile.id,
            embeddings_per_clip=count,
            error=None,
        )
        thread = threading.Thread(
            target=process_semantics_only,
            args=(
                movie_id, profile.id, count, overwrite, mode,
                adaptive_seconds, adaptive_min, adaptive_max,
            ),
            daemon=True,
        )
        _running_jobs[movie_id] = thread
        thread.start()
        return True


def is_processing(movie_id: int) -> bool:
    with _job_lock:
        thread = _running_jobs.get(movie_id)
        return bool(thread and thread.is_alive())


def running_movie_ids() -> list[int]:
    with _job_lock:
        return [movie_id for movie_id, thread in _running_jobs.items() if thread.is_alive()]


def embed_text_for_profile(profile_id: str, text: str) -> Any:
    return embed_texts_for_profile(profile_id, [text])[0]


_text_analyzer: Any = None
_text_analyzer_profile = ""


def release_text_embedding_model() -> None:
    global _text_analyzer, _text_analyzer_profile
    with _semantic_lock:
        if _text_analyzer is not None:
            _text_analyzer.close()
        _text_analyzer = None
        _text_analyzer_profile = ""


def embed_texts_for_profile(profile_id: str, texts: list[str]) -> Any:
    global _text_analyzer, _text_analyzer_profile
    profile = get_profile(profile_id)
    with _semantic_lock:
        if _text_analyzer is None or _text_analyzer_profile != profile.id:
            if _text_analyzer is not None:
                _text_analyzer.close()
            if profile.model_type == "languagebind":
                _text_analyzer = LanguageBindAnalyzer(
                    profile,
                    profile.default_embeddings_per_clip,
                )
            else:
                _text_analyzer = SemanticAnalyzer(
                    profile.model_name,
                    profile_id=profile.id,
                    embeddings_per_clip=profile.default_embeddings_per_clip,
                    input_size=profile.input_size,
                )
            _text_analyzer_profile = profile.id
        return [_text_analyzer.embed_text(text) for text in texts]


def pause_processing(movie_id: int) -> None:
    db.update_movie(movie_id, paused=1, status="paused", progress_stage="paused")


def reset_processing_outputs(movie_id: int, *, delete_source: bool = False) -> dict[str, int]:
    movie = db.get_movie(movie_id)
    if not movie:
        raise ValueError("Movie not found")

    removed = 0
    clips = db.list_clips({"movie_id": movie_id})
    for clip in clips:
        for key in ("clip_path", "metadata_path", "embedding_path"):
            value = clip.get(key)
            if value:
                path = Path(value)
                if path.exists():
                    path.unlink(missing_ok=True)
                    removed += 1
        frame_dir = FRAMES_DIR / f"clip_{int(clip['id']):06d}"
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)
            removed += 1
        profile_name = f"clip_{int(clip['id']):06d}"
        for artifact in EMBEDDING_PROFILES_DIR.glob(f"*/{profile_name}.*"):
            artifact.unlink(missing_ok=True)
            removed += 1
        for profile_frames in FRAMES_DIR.glob(f"*/{profile_name}"):
            if profile_frames.is_dir():
                shutil.rmtree(profile_frames, ignore_errors=True)
                removed += 1

    folder_name = _movie_folder_name(movie_id, movie["original_name"])
    for folder in (CLIPS_DIR / folder_name, METADATA_DIR / folder_name):
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1

    if delete_source:
        source = Path(movie["path"])
        if source.exists():
            source.unlink(missing_ok=True)
            removed += 1

    db.delete_clips_for_movie(movie_id)
    if not delete_source:
        db.update_movie(
            movie_id,
            paused=0,
            status="imported",
            progress_stage="imported",
            progress_detail="Ready to process",
            error=None,
            detector=None,
            encoder="",
            device="",
        )
    return {"files_removed": removed, "clips_removed": len(clips)}


def reset_semantic_outputs(movie_id: int) -> dict[str, int]:
    movie = db.get_movie(movie_id)
    if not movie:
        raise ValueError("Movie not found")

    removed = 0
    clips = db.list_clips({"movie_id": movie_id})
    for clip in clips:
        for key in ("metadata_path", "embedding_path"):
            value = clip.get(key)
            if value:
                path = Path(value)
                if path.exists():
                    path.unlink(missing_ok=True)
                    removed += 1
        if clip["status"] != "too_short":
            db.update_clip(
                int(clip["id"]),
                status="motion_analyzed" if clip.get("clip_path") else "detected",
                people_count="unknown",
                shot_size="unknown",
                moods=[],
                settings=[],
                quality_flags=[],
                description="",
                tags=[],
                embedding_path="",
                metadata_path="",
            )

    folder_name = _movie_folder_name(movie_id, movie["original_name"])
    metadata_dir = METADATA_DIR / folder_name
    if metadata_dir.exists():
        shutil.rmtree(metadata_dir, ignore_errors=True)
        removed += 1

    db.update_movie(
        movie_id,
        paused=0,
        status="processing",
        progress_stage="semantic_indexing",
        progress_detail="Creating semantic labels and embeddings",
        error=None,
    )
    return {"files_removed": removed, "clips_reset": sum(1 for clip in clips if clip["status"] != "too_short")}


def process_movie(movie_id: int) -> None:
    movie = db.get_movie(movie_id)
    if not movie:
        return

    pipeline_started = time.perf_counter()
    outcome = "unknown"
    timing_event("pipeline_start", movie_id=movie_id, mode="full")
    try:
        source = Path(movie["path"])
        db.update_movie(movie_id, status="processing", progress_stage="detecting_shots", progress_detail="Detecting shot boundaries")
        shots, detector = _timed_call(
            movie_id,
            "shot_detection",
            detect_shots,
            source,
            float(movie["fps"] or 25.0),
            float(movie["duration"] or 0.0),
            SETTINGS.transnet_threshold,
            SETTINGS.merge_tiny_shots_seconds,
        )
        db.update_movie(movie_id, detector=detector)
        _timed_call(movie_id, "shot_store", _store_shots, movie_id, shots)
        if _is_paused(movie_id):
            outcome = "paused"
            return

        db.update_movie(movie_id, progress_stage="exporting_clips", progress_detail="Exporting MP4 clips")
        _timed_call(movie_id, "clip_export", _export_missing_clips, movie_id, source)
        if _is_paused(movie_id):
            outcome = "paused"
            return

        db.update_movie(movie_id, progress_stage="motion_analysis", progress_detail="Analyzing camera and animation movement")
        _timed_call(movie_id, "motion_analysis", _analyze_missing_motion, movie_id, source)
        if _is_paused(movie_id):
            outcome = "paused"
            return

        db.update_movie(movie_id, progress_stage="semantic_indexing", progress_detail="Creating semantic labels and embeddings")
        _timed_call(movie_id, "semantic_indexing", _analyze_missing_semantics, movie_id, source)
        if _is_paused(movie_id):
            outcome = "paused"
            return

        db.update_movie(movie_id, progress_stage="metadata_export", progress_detail="Writing metadata sidecars")
        _timed_call(movie_id, "metadata_export", _write_all_metadata, movie_id)
        db.update_movie(movie_id, status="complete", progress_stage="complete", progress_detail="Complete", error=None)
        outcome = "complete"
    except ToolMissingError as exc:
        outcome = "error"
        db.update_movie(movie_id, status="error", progress_stage="error", error=str(exc))
    except Exception as exc:
        outcome = "error"
        db.update_movie(movie_id, status="error", progress_stage="error", error=repr(exc))
    finally:
        timing_event(
            "pipeline_end",
            movie_id=movie_id,
            mode="full",
            status=outcome,
            elapsed_s=round(time.perf_counter() - pipeline_started, 4),
        )
        with _job_lock:
            _running_jobs.pop(movie_id, None)


def process_semantics_only(
    movie_id: int,
    profile_id: str = DEFAULT_PROFILE_ID,
    embeddings_per_clip: int | None = None,
    overwrite: bool = False,
    sampling_mode: str = "adaptive",
    adaptive_seconds: float = 1.5,
    adaptive_min: int = 3,
    adaptive_max: int = 16,
) -> None:
    movie = db.get_movie(movie_id)
    if not movie:
        return

    pipeline_started = time.perf_counter()
    outcome = "unknown"
    profile = get_profile(profile_id)
    count = normalize_embeddings_per_clip(embeddings_per_clip, profile)
    timing_event(
        "pipeline_start",
        movie_id=movie_id,
        mode="semantics_only",
        profile_id=profile.id,
        embeddings_per_clip=count,
        sampling_mode=sampling_mode,
    )
    try:
        source = Path(movie["path"])
        _timed_call(
            movie_id,
            "semantic_indexing",
            _analyze_embedding_profile,
            movie_id,
            source,
            profile.id,
            count,
            overwrite,
            profile.model_type == "siglip2",
            sampling_mode,
            adaptive_seconds,
            adaptive_min,
            adaptive_max,
        )
        if _is_paused(movie_id):
            outcome = "paused"
            return
        db.update_movie(movie_id, progress_stage="metadata_export", progress_detail="Writing metadata sidecars")
        _timed_call(movie_id, "metadata_export", _write_all_metadata, movie_id)
        db.update_movie(
            movie_id,
            status="complete",
            progress_stage="complete",
            progress_detail=f"{profile.label} embeddings complete",
            active_embedding_profile=profile.id,
            embeddings_per_clip=count,
            error=None,
        )
        outcome = "complete"
    except Exception as exc:  # noqa: BLE001
        outcome = "error"
        db.update_movie(movie_id, status="error", progress_stage="error", error=repr(exc))
    finally:
        timing_event(
            "pipeline_end",
            movie_id=movie_id,
            mode="semantics_only",
            status=outcome,
            profile_id=profile.id,
            elapsed_s=round(time.perf_counter() - pipeline_started, 4),
        )
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
    profile = get_profile(DEFAULT_PROFILE_ID)
    _analyze_embedding_profile(
        movie_id,
        source,
        profile.id,
        SETTINGS.sample_frames_per_clip,
        False,
        True,
        "adaptive",
        1.5,
        3,
        16,
    )


def _saved_frame_paths(clip_id: int) -> list[str]:
    """Return previously decoded representative frames, preferring SigLIP2 224."""
    preferred = FRAMES_DIR / DEFAULT_PROFILE_ID / f"clip_{clip_id:06d}"
    candidates = sorted(preferred.glob("frame_*.jpg"))
    if candidates:
        return [str(path) for path in candidates]
    candidates = sorted(FRAMES_DIR.glob(f"*/clip_{clip_id:06d}/frame_*.jpg"))
    return [str(path) for path in candidates]


def _analyze_embedding_profile(
    movie_id: int,
    source: Path,
    profile_id: str,
    embeddings_per_clip: int,
    overwrite: bool,
    update_labels: bool,
    sampling_mode: str = "adaptive",
    adaptive_seconds: float = 1.5,
    adaptive_min: int = 3,
    adaptive_max: int = 16,
) -> None:
    profile = get_profile(profile_id)
    movie = db.get_movie(movie_id) or {}
    clips = [clip for clip in db.list_clips({"movie_id": movie_id}) if clip["status"] != "too_short"]
    analyzer: SemanticAnalyzer | LanguageBindAnalyzer
    db.update_movie(
        movie_id,
        progress_stage="semantic_indexing",
        progress_detail=f"Queued for GPU slot: {profile.label}",
    )
    with _semantic_queue_slot(movie_id, profile.label) as queue_wait_s, _semantic_lock:
        timing_event(
            "semantic_gpu_acquired",
            movie_id=movie_id,
            profile_id=profile.id,
            queue_wait_s=round(queue_wait_s, 4),
        )
        db.update_movie(
            movie_id,
            progress_stage="semantic_indexing",
            progress_detail=f"{profile.label}: starting",
        )
        release_text_embedding_model()
        if _is_paused(movie_id):
            return
        if profile.model_type == "languagebind":
            analyzer = LanguageBindAnalyzer(profile, embeddings_per_clip)
        else:
            analyzer = SemanticAnalyzer(
                profile.model_name,
                profile_id=profile.id,
                embeddings_per_clip=embeddings_per_clip,
                input_size=profile.input_size,
            )
        try:
            for position, clip in enumerate(clips, start=1):
                if _is_paused(movie_id):
                    return
                clip_id = int(clip["id"])
                clip_embeddings = (
                    adaptive_embeddings_per_clip(
                        float(clip.get("duration") or 0),
                        seconds_per_vector=adaptive_seconds,
                        minimum=adaptive_min,
                        maximum=adaptive_max,
                    )
                    if sampling_mode != "fixed"
                    else embeddings_per_clip
                )
                analyzer.embeddings_per_clip = clip_embeddings
                existing = db.get_clip_embedding(clip_id, profile.id)
                if (
                    not overwrite
                    and existing
                    and existing.get("status") == "complete"
                    and int(existing.get("frame_count") or 0) == clip_embeddings
                    and Path(existing.get("artifact_path") or "").exists()
                ):
                    continue
                db.update_movie(
                    movie_id,
                    progress_detail=f"{profile.label}: {position}/{len(clips)}",
                    active_embedding_profile=profile.id,
                    embeddings_per_clip=clip_embeddings,
                )
                clip_started = time.perf_counter()
                try:
                    saved_frames = _saved_frame_paths(clip_id)
                    analyze_kwargs = {"movie_id": movie_id}
                    if profile.model_type == "siglip2":
                        analyze_kwargs["existing_frame_paths"] = saved_frames
                    result = analyzer.analyze_clip(
                        source,
                        clip_id,
                        float(clip["start_time"]),
                        float(clip["end_time"]),
                        **analyze_kwargs,
                    )
                    timings = result.pop("_timings", {})
                    if result.get("semantic_model") == "fallback-cv":
                        raise RuntimeError(result.get("description") or "Semantic model unavailable")
                    db.upsert_clip_embedding(
                        clip_id,
                        profile.id,
                        artifact_path=result["embedding_path"],
                        frame_count=int(result.get("embedding_count") or 0),
                        dimension=int(result.get("embedding_dimension") or 0),
                        status="complete",
                        source_checksum=str(movie.get("checksum") or ""),
                    )
                    db_started = time.perf_counter()
                    if update_labels and "tags" in result:
                        tags = result["tags"]
                        if result.get("semantic_model"):
                            tags = list(dict.fromkeys([*tags, result["semantic_model"]]))
                        db.update_clip(
                            clip_id,
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
                    db_update_s = time.perf_counter() - db_started
                    metadata_started = time.perf_counter()
                    _write_metadata(
                        movie_id,
                        clip_id,
                        extra={
                            "representative_frames": result.get("frame_paths", []),
                            "embedding_profile": profile.to_dict(),
                            "embeddings_per_clip": clip_embeddings,
                            "sampling_mode": sampling_mode,
                            "adaptive_sampling": {
                                "seconds_per_vector": adaptive_seconds,
                                "minimum": adaptive_min,
                                "maximum": adaptive_max,
                            } if sampling_mode != "fixed" else None,
                        },
                    )
                    metadata_s = time.perf_counter() - metadata_started
                    timing_event(
                        "semantic_clip",
                        movie_id=movie_id,
                        clip_id=clip_id,
                        clip_index=int(clip["clip_index"]),
                        profile_id=profile.id,
                        embeddings_per_clip=clip_embeddings,
                        sampling_mode=sampling_mode,
                        start_time=round(float(clip["start_time"]), 3),
                        end_time=round(float(clip["end_time"]), 3),
                        db_update_s=round(db_update_s, 4),
                        metadata_s=round(metadata_s, 4),
                        total_s=round(time.perf_counter() - clip_started, 4),
                        **timings,
                    )
                except Exception as exc:  # noqa: BLE001
                    db.upsert_clip_embedding(
                        clip_id,
                        profile.id,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        source_checksum=str(movie.get("checksum") or ""),
                    )
                    raise
        finally:
            analyzer.close()


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
