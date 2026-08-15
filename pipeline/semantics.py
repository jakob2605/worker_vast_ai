from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import EMBEDDING_PROFILES_DIR, FRAMES_DIR, SETTINGS
from .motion import sample_frames
from .profiles import DEFAULT_PROFILE_ID


PEOPLE_LABELS = {
    "none": ["empty scene", "landscape with no people", "room with no person"],
    "one": ["one person", "single character", "portrait of one person"],
    "two": ["two people", "two characters"],
    "group": ["group of people", "crowd", "many characters"],
}
SHOT_SIZE_LABELS = ["close-up face", "medium shot person", "wide shot environment"]
MOOD_LABELS = ["calm", "tense", "sad", "lonely", "romantic", "dark", "bright", "chaotic", "joyful"]
SETTING_LABELS = ["indoor", "outdoor", "city", "nature", "room", "vehicle", "school", "restaurant", "night", "day"]


def _load_saved_frames(paths: list[str]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for value in paths:
        frame = cv2.imread(str(value))
        if frame is not None:
            frames.append(frame)
    return frames


class SemanticAnalyzer:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        profile_id: str = DEFAULT_PROFILE_ID,
        embeddings_per_clip: int | None = None,
        input_size: int = 384,
    ) -> None:
        self.model_name = model_name or SETTINGS.siglip_model
        self.profile_id = profile_id
        self.embeddings_per_clip = embeddings_per_clip or SETTINGS.sample_frames_per_clip
        self.input_size = input_size
        self._loaded = False
        self._available = False
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str = SETTINGS.device
        self._dtype: Any = None
        self._last_error = ""

    def analyze_clip(
        self,
        video_path: Path,
        clip_id: int,
        start_time: float,
        end_time: float,
        *,
        movie_id: int | None = None,
        existing_frame_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        decode_started = time.perf_counter()
        frames = _load_saved_frames(existing_frame_paths or [])
        if not frames:
            frames = sample_frames(video_path, start_time, end_time, self.embeddings_per_clip, self.input_size)
        frame_times = _sample_times(start_time, end_time, len(frames))
        decode_s = time.perf_counter() - decode_started
        frame_save_started = time.perf_counter()
        frame_paths = self._save_representative_frames(frames, clip_id)
        frame_save_s = time.perf_counter() - frame_save_started
        model_load_started = time.perf_counter()
        model_available = bool(frames) and self._ensure_model()
        model_load_s = time.perf_counter() - model_load_started
        if not frames:
            result = self._fallback_result([], clip_id, frame_paths)
        elif model_available:
            try:
                result = self._siglip_result(frames, frame_times, clip_id, frame_paths)
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"SigLIP inference failed: {type(exc).__name__}: {exc}"
                result = self._fallback_result(frames, clip_id, frame_paths)
        else:
            result = self._fallback_result(frames, clip_id, frame_paths)

        timings = result.pop("_timings", {})
        result["_timings"] = {
            "frames": len(frames),
            "device": self._device,
            "model": result.get("semantic_model", ""),
            "fallback": result.get("semantic_model") == "fallback-cv",
            "decode_s": round(decode_s, 4),
            "frame_save_s": round(frame_save_s, 4),
            "model_load_s": round(model_load_s, 4),
            "analysis_s": round(time.perf_counter() - total_started, 4),
            **timings,
        }
        return result

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
            "profile_id": self.profile_id,
            "loaded": self._loaded,
            "available": self._available,
            "device": getattr(self, "_device", SETTINGS.device),
            "fp16": SETTINGS.siglip_fp16,
            "batch_size": SETTINGS.siglip_batch_size,
            "embeddings_per_clip": self.embeddings_per_clip,
            "error": self._last_error,
        }

    def embed_text(self, text: str) -> np.ndarray:
        if not self._ensure_model():
            raise RuntimeError(self._last_error or "SigLIP model is unavailable")
        inputs = self._processor(text=[text], padding="max_length", return_tensors="pt")
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self._torch.no_grad():
            if hasattr(self._model, "get_text_features"):
                try:
                    text_embeds = self._model.get_text_features(**inputs)
                except TypeError:
                    allowed = {"input_ids", "attention_mask", "position_ids"}
                    text_embeds = self._model.get_text_features(**{k: v for k, v in inputs.items() if k in allowed})
            else:
                outputs = self._model(**inputs)
                text_embeds = outputs.text_embeds
            text_embeds = text_embeds.float()
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds[0].cpu().numpy().astype("float32")

    def _siglip_result(
        self,
        frames: list[np.ndarray],
        frame_times: list[float],
        clip_id: int,
        frame_paths: list[str],
    ) -> dict[str, Any]:
        preprocess_started = time.perf_counter()
        images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        preprocess_s = time.perf_counter() - preprocess_started
        text_labels = _all_text_labels()
        processor_started = time.perf_counter()
        inputs = self._processor(text=text_labels, images=images, padding="max_length", return_tensors="pt")
        processor_s = time.perf_counter() - processor_started
        # Tensors must live on the same device as the model.
        transfer_started = time.perf_counter()
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if self._dtype == self._torch.float16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        self._sync_device()
        transfer_s = time.perf_counter() - transfer_started

        model_started = time.perf_counter()
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        self._sync_device()
        model_forward_s = time.perf_counter() - model_started

        embedding_started = time.perf_counter()
        with self._torch.no_grad():
            image_embeds = outputs.image_embeds.float()
            text_embeds = outputs.text_embeds.float()
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            similarities = image_embeds @ text_embeds.T
            mean_scores = similarities.mean(dim=0).cpu().numpy()
            mean_embedding = image_embeds.mean(dim=0)
            mean_embedding = mean_embedding / mean_embedding.norm().clamp_min(1e-9)
            frame_embeddings = image_embeds.cpu().numpy().astype("float32")
            embedding = mean_embedding.cpu().numpy().astype("float32")
        embedding_reduce_s = time.perf_counter() - embedding_started

        artifact_started = time.perf_counter()
        scored = sorted(zip(text_labels, mean_scores), key=lambda item: float(item[1]), reverse=True)
        embedding_path = EMBEDDING_PROFILES_DIR / self.profile_id / f"clip_{clip_id:06d}.npz"
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            embedding_path,
            mean=embedding,
            frames=frame_embeddings,
            frame_times=np.asarray(frame_times, dtype="float32"),
            model=np.asarray(self.model_name),
            profile_id=np.asarray(self.profile_id),
            normalized=np.asarray(True),
        )

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
            "embedding_count": int(frame_embeddings.shape[0]),
            "embedding_dimension": int(frame_embeddings.shape[1]),
            "frame_paths": frame_paths,
            "semantic_model": self.model_name,
            "_timings": {
                "preprocess_s": round(preprocess_s, 4),
                "processor_s": round(processor_s, 4),
                "transfer_s": round(transfer_s, 4),
                "model_forward_s": round(model_forward_s, 4),
                "embedding_reduce_s": round(embedding_reduce_s, 4),
                "artifact_s": round(time.perf_counter() - artifact_started, 4),
            },
        }

    def _sync_device(self) -> None:
        if self._device == "cuda" and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

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
        embedding_path = EMBEDDING_PROFILES_DIR / self.profile_id / f"clip_{clip_id:06d}.json"
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
        clip_dir = FRAMES_DIR / self.profile_id / f"clip_{clip_id:06d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in clip_dir.glob("*.jpg"):
            old_frame.unlink(missing_ok=True)
        paths: list[str] = []
        for index, frame in enumerate(frames):
            path = clip_dir / f"frame_{index + 1}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            paths.append(str(path))
        return paths

    def close(self) -> None:
        self._processor = None
        self._model = None
        if self._torch is not None and self._device == "cuda" and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def classify_clip_shot_size(clip_path: Path, duration: float) -> str:
    frames = sample_frames(clip_path, 0.0, max(0.01, duration), SETTINGS.sample_frames_per_clip, 384)
    return _heuristic_shot_size(frames)


def _sample_times(start_time: float, end_time: float, count: int) -> list[float]:
    if count <= 0:
        return []
    duration = max(0.01, end_time - start_time)
    return [
        round(float(value), 4)
        for value in np.linspace(start_time + duration * 0.12, end_time - duration * 0.12, count)
    ]


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
