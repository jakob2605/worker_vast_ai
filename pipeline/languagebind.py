from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import EMBEDDING_PROFILES_DIR, FRAMES_DIR, ROOT
from .motion import sample_frames
from .profiles import EmbeddingProfile


class LanguageBindAnalyzer:
    def __init__(self, profile: EmbeddingProfile, embeddings_per_clip: int) -> None:
        self.profile = profile
        self.embeddings_per_clip = embeddings_per_clip
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        python = os.getenv("LANGUAGEBIND_PYTHON", "/workspace/venvs/languagebind/bin/python")
        script = ROOT / "tools" / "languagebind_worker.py"
        if not Path(python).exists():
            raise RuntimeError(
                f"LanguageBind runtime is missing at {python}. Run bootstrap_languagebind.sh on the worker."
            )
        env = dict(os.environ)
        env["LB_MODEL"] = self.profile.model_name
        env["LB_FRAMES"] = str(self.profile.frames_per_embedding)
        self._proc = subprocess.Popen(
            [python, str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
            bufsize=1,
        )
        return self._proc

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            proc = self._start()
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                error = proc.stderr.read()[-2000:] if proc.stderr else ""
                self._proc = None
                raise RuntimeError(f"LanguageBind worker stopped unexpectedly. {error}")
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "LanguageBind request failed")
        return result

    @staticmethod
    def _vectors(result: dict[str, Any]) -> np.ndarray:
        raw = base64.b64decode(result["data"])
        return np.frombuffer(raw, dtype="float32").reshape(result["shape"])

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
        del movie_id
        duration = max(0.01, end_time - start_time)
        boundaries = np.linspace(start_time, end_time, self.embeddings_per_clip + 1)
        vectors: list[np.ndarray] = []
        times: list[float] = []
        saved_paths: list[str] = []
        clip_dir = FRAMES_DIR / self.profile.id / f"clip_{clip_id:06d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in clip_dir.glob("*.jpg"):
            old_frame.unlink(missing_ok=True)

        for window_index in range(self.embeddings_per_clip):
            lo = float(boundaries[window_index])
            hi = float(boundaries[window_index + 1])
            frames = sample_frames(
                video_path,
                lo,
                hi,
                self.profile.frames_per_embedding,
                self.profile.input_size,
            )
            paths: list[Path] = []
            for frame_index, frame in enumerate(frames):
                path = clip_dir / f"window_{window_index + 1:02d}_frame_{frame_index + 1:02d}.jpg"
                cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                paths.append(path)
                saved_paths.append(str(path))
            # Reuse frames produced earlier in the pipeline if the source
            # cannot be reopened/seeked by OpenCV (common with mounted files).
            if not paths and existing_frame_paths:
                start = round(window_index * len(existing_frame_paths) / self.embeddings_per_clip)
                stop = round((window_index + 1) * len(existing_frame_paths) / self.embeddings_per_clip)
                paths = [Path(path) for path in existing_frame_paths[start:stop]]
            if not paths:
                continue
            result = self._request({"op": "images", "paths": [str(path) for path in paths]})
            encoded = self._vectors(result)
            if encoded.size:
                vectors.append(encoded[0])
                times.append(round((lo + hi) / 2.0, 4))

        if not vectors:
            raise RuntimeError("LanguageBind produced no clip embeddings")
        frame_vectors = _normalize(np.vstack(vectors).astype("float32"))
        mean = _normalize(frame_vectors.mean(axis=0, keepdims=True))[0]
        artifact = EMBEDDING_PROFILES_DIR / self.profile.id / f"clip_{clip_id:06d}.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            mean=mean,
            frames=frame_vectors,
            frame_times=np.asarray(times, dtype="float32"),
            model=np.asarray(self.profile.model_name),
            profile_id=np.asarray(self.profile.id),
            normalized=np.asarray(True),
            video=np.asarray(True),
        )
        return {
            "embedding_path": str(artifact),
            "embedding_count": int(frame_vectors.shape[0]),
            "embedding_dimension": int(frame_vectors.shape[1]),
            "frame_paths": saved_paths,
            "semantic_model": self.profile.model_name,
            "_timings": {"frames": len(saved_paths), "model": self.profile.model_name},
        }

    def embed_text(self, text: str) -> np.ndarray:
        result = self._request({"op": "texts", "texts": [text]})
        vectors = self._vectors(result)
        if not vectors.size:
            raise RuntimeError("LanguageBind produced no text embedding")
        return _normalize(vectors)[0]

    def device_info(self) -> dict[str, Any]:
        try:
            result = self._request({"op": "ping"})
            return {**result, "profile_id": self.profile.id, "available": True}
        except Exception as exc:  # noqa: BLE001
            return {"profile_id": self.profile.id, "available": False, "error": str(exc)}

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._request({"op": "quit"})
            except Exception:  # noqa: BLE001
                self._proc.kill()
        self._proc = None


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-9)).astype("float32")
