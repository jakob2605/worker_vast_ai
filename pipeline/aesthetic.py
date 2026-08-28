"""Aesthetic Predictor V2.5 scoring for the cleanup/review workflow."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from .config import FRAMES_DIR
from . import db

MODEL_VERSION = "aesthetic-predictor-v2-5"
_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
_PREPROCESSOR: Any = None
_TORCH: Any = None


def _predictor(progress: Callable[[str, float], None] | None = None) -> tuple[Any, Any, Any]:
    global _MODEL, _PREPROCESSOR, _TORCH
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _PREPROCESSOR, _TORCH
        if progress:
            progress("Aesthetic-Modell wird geladen …", 0.0)
        try:
            import torch
            from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        except ImportError as exc:
            raise RuntimeError(
                "Aesthetic Predictor V2.5 is not installed. Install "
                "aesthetic-predictor-v2-5, torch, transformers, pillow and opencv-python-headless."
            ) from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model, preprocessor = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)
        _MODEL = model.to(device=device, dtype=dtype).eval()
        _PREPROCESSOR = preprocessor
        _TORCH = torch
        if progress:
            progress("Aesthetic-Modell bereit", 0.0)
        return _MODEL, _PREPROCESSOR, _TORCH


def _sample_images(clip_id: int) -> list[Any]:
    import cv2

    # The worker usually already has representative frames for embedding
    # profiles. Reading those is substantially faster than seeking through
    # every MP4 again, and it uses the same visual samples consistently.
    frame_dirs = [
        FRAMES_DIR / "siglip2-base-224" / f"clip_{clip_id:06d}",
        FRAMES_DIR / f"clip_{clip_id:06d}",
    ]
    frame_dirs.extend(sorted(FRAMES_DIR.glob(f"*/clip_{clip_id:06d}")))
    for frame_dir in frame_dirs:
        if not frame_dir.is_dir():
            continue
        paths = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if paths:
            images = []
            for image_path in paths:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is not None:
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return images
    return []


def _score(images: list[Any], predictor: tuple[Any, Any, Any] | None = None) -> list[float]:
    if not images:
        return []
    model, preprocessor, torch = predictor or _predictor()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    batch = preprocessor(images=images, return_tensors="pt")
    pixels = batch.pixel_values.to(device=device, dtype=dtype)
    with torch.inference_mode():
        return [float(value) for value in model(pixels).logits.reshape(-1).float().cpu().tolist()]


def load_scores(movie_id: int | None = None) -> dict[int, dict[str, Any]]:
    return {int(row["clip_id"]): row for row in db.list_aesthetic_scores(movie_id)}


def score_movie(movie_id: int, *, sample_frames: int = 0, overwrite: bool = False, progress: Callable[[str, float], None] | None = None) -> dict[str, Any]:
    clips = db.list_clips({"movie_id": int(movie_id), "has_file": True})
    scores = load_scores(movie_id)
    scored = 0
    skipped = 0
    total = len(clips)
    pending = [clip for clip in clips if overwrite or int(clip["id"]) not in scores]
    existing = total - len(pending)
    if progress:
        progress(f"{total} Clips gefunden; {existing} bereits bewertet", 0.0)
    if not pending:
        return {"movie_id": int(movie_id), "scored": 0, "skipped": 0,
                "existing": existing, "total": total}
    predictor = _predictor(progress)
    batch_size = max(1, min(16, int(os.getenv("AESTHETIC_BATCH_CLIPS", "8"))))
    for batch_start in range(0, len(pending), batch_size):
        batch_clips = pending[batch_start:batch_start + batch_size]
        images: list[Any] = []
        image_counts: list[int] = []
        for clip in batch_clips:
            clip_images = _sample_images(int(clip["id"]))
            images.extend(clip_images)
            image_counts.append(len(clip_images))
        values = _score(images, predictor)
        cursor = 0
        for clip, image_count in zip(batch_clips, image_counts):
            clip_values = values[cursor:cursor + image_count]
            cursor += image_count
            if not clip_values:
                skipped += 1
                continue
            row = {
                "clip_id": int(clip["id"]), "movie_id": int(movie_id),
                "model_version": MODEL_VERSION,
                "sampled_scores": [round(value, 5) for value in clip_values],
                "mean_score": round(sum(clip_values) / len(clip_values), 5),
                "min_score": round(min(clip_values), 5), "max_score": round(max(clip_values), 5),
                "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            db.upsert_aesthetic_score(
                int(clip["id"]), model_version=MODEL_VERSION,
                sampled_scores=row["sampled_scores"], mean_score=row["mean_score"],
                min_score=row["min_score"], max_score=row["max_score"], scored_at=row["scored_at"],
            )
            scores[int(clip["id"])] = row
            scored += 1
        completed = min(len(pending), batch_start + len(batch_clips))
        if progress:
            progress(f"Aesthetic: {completed}/{len(pending)}", completed / max(1, len(pending)))
    return {"movie_id": int(movie_id), "scored": scored, "skipped": skipped,
            "existing": existing, "total": total}


def score_all_movies(*, sample_frames: int = 0, overwrite: bool = False,
                     progress: Callable[[str, float], None] | None = None) -> dict[str, Any]:
    """Backfill every movie using only its already-saved representative frames."""
    movies = [movie for movie in db.list_movies()
              if db.count_clips({"movie_id": int(movie["id"]), "has_file": True})]
    aggregate = {"movies": len(movies), "scored": 0, "skipped": 0,
                 "existing": 0, "total": 0}
    if not movies:
        if progress:
            progress("Keine Clips zum Bewerten gefunden", 1.0)
        return aggregate
    for index, movie in enumerate(movies):
        title = str(movie.get("collection_title") or movie.get("original_name") or movie["id"])

        def movie_progress(message: str, value: float) -> None:
            if progress:
                progress(f"Film {index + 1}/{len(movies)}: {title} – {message}",
                         (index + max(0.0, min(1.0, value))) / len(movies))

        result = score_movie(int(movie["id"]), sample_frames=sample_frames,
                             overwrite=overwrite, progress=movie_progress)
        for key in ("scored", "skipped", "existing", "total"):
            aggregate[key] += int(result.get(key) or 0)
    if progress:
        progress(f"Aesthetic abgeschlossen: {aggregate['scored']} neue Scores", 1.0)
    return aggregate


def delete_scores(clip_ids: list[int]) -> int:
    return db.delete_aesthetic_scores(clip_ids)
