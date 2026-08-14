from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import EMBEDDINGS_DIR, FRAMES_DIR, SETTINGS
from .motion import sample_frames


PEOPLE_LABELS = {
    "none": ["empty scene", "landscape with no people", "room with no person"],
    "one": ["one person", "single character", "portrait of one person"],
    "two": ["two people", "two characters"],
    "group": ["group of people", "crowd", "many characters"],
}
SHOT_SIZE_LABELS = ["close-up face", "medium shot person", "wide shot environment"]
MOOD_LABELS = ["calm", "tense", "sad", "lonely", "romantic", "dark", "bright", "chaotic", "joyful"]
SETTING_LABELS = ["indoor", "outdoor", "city", "nature", "room", "vehicle", "school", "restaurant", "night", "day"]


class SemanticAnalyzer:
    def __init__(self) -> None:
        self.model_name = SETTINGS.siglip_model
        self._loaded = False
        self._available = False
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str = SETTINGS.device
        self._dtype: Any = None
        self._last_error = ""

    def analyze_clip(self, video_path: Path, clip_id: int, start_time: float, end_time: float) -> dict[str, Any]:
        frames = sample_frames(video_path, start_time, end_time, SETTINGS.sample_frames_per_clip, 384)
        frame_paths = self._save_representative_frames(frames, clip_id)
        if not frames:
            return self._fallback_result([], clip_id, frame_paths)
        if self._ensure_model():
            try:
                return self._siglip_result(frames, clip_id, frame_paths)
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"SigLIP inference failed: {type(exc).__name__}: {exc}"
        return self._fallback_result(frames, clip_id, frame_paths)

    def _ensure_model(self) -> bool:
        if self._loaded:
            return self._available
        self._loaded = True
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._torch = torch
            self._device = SETTINGS.device
            self._dtype = torch.float16 if (SETTINGS.siglip_fp16 and self._device == "cuda") else torch.float32
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name, torch_dtype=self._dtype)
            # The original never left the CPU. Moving the model is the second GPU win.
            self._model = model.to(self._device)
            self._model.eval()
            self._available = True
            self._last_error = ""
        except Exception as exc:  # noqa: BLE001
            self._available = False
            self._last_error = f"SigLIP unavailable: {type(exc).__name__}: {exc}"
        return self._available

    def device_info(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "loaded": self._loaded,
            "available": self._available,
            "device": getattr(self, "_device", SETTINGS.device),
            "fp16": SETTINGS.siglip_fp16,
            "batch_size": SETTINGS.siglip_batch_size,
            "error": self._last_error,
        }

    def _siglip_result(self, frames: list[np.ndarray], clip_id: int, frame_paths: list[str]) -> dict[str, Any]:
        images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        text_labels = _all_text_labels()
        inputs = self._processor(text=text_labels, images=images, padding="max_length", return_tensors="pt")
        # Tensors must live on the same device as the model.
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if self._dtype == self._torch.float16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            image_embeds = outputs.image_embeds.float()
            text_embeds = outputs.text_embeds.float()
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            similarities = image_embeds @ text_embeds.T
            mean_scores = similarities.mean(dim=0).cpu().numpy()
            embedding = image_embeds.mean(dim=0).cpu().numpy().astype("float32")

        scored = sorted(zip(text_labels, mean_scores), key=lambda item: float(item[1]), reverse=True)
        embedding_path = EMBEDDINGS_DIR / f"clip_{clip_id:06d}.npy"
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, embedding)

        people_count = _best_people(scored)
        shot_size = _best_from(scored, SHOT_SIZE_LABELS, {"close-up face": "close_up", "medium shot person": "medium_shot", "wide shot environment": "wide_shot"})
        if shot_size == "unknown":
            shot_size = _heuristic_shot_size(frames)
        moods = [label for label, _ in scored if label in MOOD_LABELS][:3]
        settings = [label for label, _ in scored if label in SETTING_LABELS][:4]
        tags = list(dict.fromkeys([people_count, shot_size, *moods, *settings]))
        description = _description(people_count, shot_size, moods, settings)

        return {
            "people_count": people_count,
            "shot_size": shot_size,
            "moods": moods,
            "settings": settings,
            "quality_flags": _quality_flags(frames),
            "tags": [tag for tag in tags if tag and tag != "unknown"],
            "description": description,
            "embedding_path": str(embedding_path),
            "frame_paths": frame_paths,
            "semantic_model": self.model_name,
        }

    def _fallback_result(self, frames: list[np.ndarray], clip_id: int, frame_paths: list[str]) -> dict[str, Any]:
        quality = _quality_flags(frames)
        brightness = float(np.mean([np.mean(frame) for frame in frames])) if frames else 0.0
        moods = ["dark"] if brightness < 75 else ["bright"] if brightness > 170 else ["calm"]
        shot_size = _heuristic_shot_size(frames)
        tags = [shot_size, *moods, "low_confidence"]
        if "low_confidence" not in quality:
            quality.append("low_confidence")
        if "siglip_unavailable" not in quality:
            quality.append("siglip_unavailable")
        embedding_path = EMBEDDINGS_DIR / f"clip_{clip_id:06d}.json"
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_path.write_text(
            json.dumps({"fallback": True, "brightness": brightness, "error": self._last_error}),
            encoding="utf-8",
        )
        description = self._last_error or "Fallback visual analysis; SigLIP2 semantic model was unavailable."
        return {
            "people_count": "unknown",
            "shot_size": shot_size,
            "moods": moods,
            "settings": [],
            "quality_flags": quality,
            "tags": tags,
            "description": description,
            "embedding_path": str(embedding_path),
            "frame_paths": frame_paths,
            "semantic_model": "fallback-cv",
        }

    def _save_representative_frames(self, frames: list[np.ndarray], clip_id: int) -> list[str]:
        clip_dir = FRAMES_DIR / f"clip_{clip_id:06d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for index, frame in enumerate(frames):
            path = clip_dir / f"frame_{index + 1}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            paths.append(str(path))
        return paths


def classify_clip_shot_size(clip_path: Path, duration: float) -> str:
    frames = sample_frames(clip_path, 0.0, max(0.01, duration), SETTINGS.sample_frames_per_clip, 384)
    return _heuristic_shot_size(frames)


def _all_text_labels() -> list[str]:
    people = [label for labels in PEOPLE_LABELS.values() for label in labels]
    return people + SHOT_SIZE_LABELS + MOOD_LABELS + SETTING_LABELS


def _best_people(scored: list[tuple[str, float]]) -> str:
    best_bucket = "unknown"
    best_score = -999.0
    for bucket, labels in PEOPLE_LABELS.items():
        score = max((float(value) for label, value in scored if label in labels), default=-999.0)
        if score > best_score:
            best_bucket = bucket
            best_score = score
    return best_bucket


def _best_from(scored: list[tuple[str, float]], labels: list[str], mapping: dict[str, str]) -> str:
    for label, _ in scored:
        if label in labels:
            return mapping.get(label, label)
    return "unknown"


def _quality_flags(frames: list[np.ndarray]) -> list[str]:
    flags: list[str] = []
    if not frames:
        return ["low_confidence"]
    brightness = float(np.mean([np.mean(frame) for frame in frames]))
    blur = float(np.mean([cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for frame in frames]))
    if brightness < 12:
        flags.append("black")
    if blur < 25:
        flags.append("blurry")
    return flags


def _heuristic_shot_size(frames: list[np.ndarray]) -> str:
    if not frames:
        return "unknown"

    concentrations: list[float] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 135)
        height, width = edges.shape
        center = edges[int(height * 0.18) : int(height * 0.82), int(width * 0.18) : int(width * 0.82)]
        full_density = float(np.count_nonzero(edges)) / max(1, edges.size)
        center_density = float(np.count_nonzero(center)) / max(1, center.size)
        concentration = center_density / max(0.001, full_density)
        concentrations.append(concentration)

    score = float(np.median(concentrations))
    if score >= 1.55:
        return "close_up"
    if score >= 1.12:
        return "medium_shot"
    return "wide_shot"


def _description(people: str, shot_size: str, moods: list[str], settings: list[str]) -> str:
    parts = []
    if people != "unknown":
        parts.append(f"{people} people")
    if shot_size != "unknown":
        parts.append(shot_size.replace("_", " "))
    if settings:
        parts.append(", ".join(settings[:2]))
    if moods:
        parts.append("mood: " + ", ".join(moods[:2]))
    return "; ".join(parts) if parts else "Semantic clip analysis"
