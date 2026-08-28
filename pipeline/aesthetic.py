"""Aesthetic Predictor V2.5 scoring for the cleanup/review workflow."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import FRAMES_DIR, LIBRARY_DIR
from . import db

SCORES_PATH = Path(os.getenv("AESTHETIC_INDEX_PATH", LIBRARY_DIR / "aesthetic_index.jsonl"))
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


def _sample_images(path: Path, count: int, clip_id: int = 0) -> list[Any]:
    import cv2

    # The worker usually already has representative frames for embedding
    # profiles. Reading those is substantially faster than seeking through
    # every MP4 again, and it uses the same visual samples consistently.
    frame_dirs = []
    if clip_id:
        frame_dirs.extend([
            FRAMES_DIR / "siglip2-base-224" / f"clip_{clip_id:06d}",
            FRAMES_DIR / f"clip_{clip_id:06d}",
        ])
        frame_dirs.extend(sorted(FRAMES_DIR.glob(f"*/clip_{clip_id:06d}")))
    for frame_dir in frame_dirs:
        if not frame_dir.is_dir():
            continue
        paths = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if paths:
            indexes = sorted({min(len(paths) - 1, int(round(i * (len(paths) - 1) / max(1, count - 1)))) for i in range(count)})
            images = []
            for index in indexes:
                image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
                if image is not None:
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return images

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        indices = sorted({min(total - 1, int(round(i * (total - 1) / max(1, count - 1)))) for i in range(count)})
        images = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return images
    finally:
        capture.release()


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


def load_scores() -> dict[int, dict[str, Any]]:
    if not SCORES_PATH.is_file():
        return {}
    result: dict[int, dict[str, Any]] = {}
    with SCORES_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                clip_id = int(row.get("clip_id") or 0)
                if clip_id:
                    result[clip_id] = row
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return result


def _save_scores(rows: dict[int, dict[str, Any]]) -> None:
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SCORES_PATH.with_suffix(SCORES_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(rows.values(), key=lambda item: int(item.get("clip_id") or 0)):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, SCORES_PATH)


def score_movie(movie_id: int, *, sample_frames: int = 8, progress: Callable[[str, float], None] | None = None) -> dict[str, Any]:
    clips = db.list_clips({"movie_id": int(movie_id), "has_file": True})
    scores = load_scores()
    scored = 0
    skipped = 0
    total = len(clips)
    if progress:
        progress(f"{total} Clips gefunden; Modell wird vorbereitet …", 0.0)
    predictor = _predictor(progress)
    batch_size = max(1, min(16, int(os.getenv("AESTHETIC_BATCH_CLIPS", "8"))))
    for batch_start in range(0, total, batch_size):
        batch_clips = clips[batch_start:batch_start + batch_size]
        images: list[Any] = []
        image_counts: list[int] = []
        for clip in batch_clips:
            path = Path(clip.get("clip_path") or "")
            clip_images = _sample_images(path, max(1, min(32, int(sample_frames))), int(clip["id"])) if path.is_file() else []
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
            scores[int(clip["id"])] = {
                "clip_id": int(clip["id"]), "movie_id": int(movie_id),
                "sampled_scores": [round(value, 5) for value in clip_values],
                "mean_score": round(sum(clip_values) / len(clip_values), 5),
                "min_score": round(min(clip_values), 5), "max_score": round(max(clip_values), 5),
                "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            scored += 1
        completed = min(total, batch_start + len(batch_clips))
        if progress:
            progress(f"Aesthetic: {completed}/{total}", completed / max(1, total))
    _save_scores(scores)
    return {"movie_id": int(movie_id), "scored": scored, "skipped": skipped, "total": total, "path": str(SCORES_PATH)}


def delete_scores(clip_ids: list[int]) -> int:
    scores = load_scores()
    removed = sum(1 for clip_id in clip_ids if scores.pop(int(clip_id), None) is not None)
    if removed:
        _save_scores(scores)
    return removed
