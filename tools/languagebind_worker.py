from __future__ import annotations

import base64
import importlib
import json
import os
import sys
import traceback
import types
from pathlib import Path

import numpy as np

MODEL_NAME = os.getenv("LB_MODEL", "LanguageBind/LanguageBind_Video_V1.5_FT")
FRAME_COUNT = int(os.getenv("LB_FRAMES", "8"))
REPO = Path(os.getenv("LANGUAGEBIND_REPO", "/workspace/LanguageBind"))


def _model_classes():
    root = REPO / "languagebind"
    if not (root / "video" / "modeling_video.py").exists():
        raise ImportError(f"LanguageBind repository not found at {REPO}")
    for name, folder in (("languagebind", root), ("languagebind.video", root / "video")):
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        module.__package__ = name
        sys.modules[name] = module
    modeling = importlib.import_module("languagebind.video.modeling_video")
    tokenizing = importlib.import_module("languagebind.video.tokenization_video")
    return modeling.LanguageBindVideo, tokenizing.LanguageBindVideoTokenizer


class Runtime:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = "cpu"
        self.dimension = 0
        self.frame_count = FRAME_COUNT

    def load(self) -> None:
        if self.model is not None:
            return
        import torch

        model_class, tokenizer_class = _model_classes()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = model_class.from_pretrained(MODEL_NAME)
        self.model = model.to(device=self.device, dtype=dtype).eval()
        self.tokenizer = tokenizer_class.from_pretrained(MODEL_NAME)
        self.torch = torch
        self.dimension = int(getattr(model.config, "projection_dim", 0) or 768)
        configured = int(getattr(model.config.vision_config, "num_frames", 0) or 0)
        if configured:
            self.frame_count = configured

    def texts(self, texts: list[str]) -> np.ndarray:
        self.load()
        encoded = self.tokenizer(
            texts,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.get_text_features(**encoded)
        return _normalize(output.float().cpu().numpy())

    def images(self, paths: list[str]) -> np.ndarray:
        self.load()
        from PIL import Image

        images = [Image.open(path).convert("RGB") for path in paths]
        if not images:
            return np.zeros((0, self.dimension), dtype="float32")
        if len(images) < self.frame_count:
            images.extend([images[-1]] * (self.frame_count - len(images)))
        elif len(images) > self.frame_count:
            step = len(images) / self.frame_count
            images = [images[int(index * step)] for index in range(self.frame_count)]
        pixels = self._pixels(images).to(self.device)
        if self.device == "cuda":
            pixels = pixels.half()
        with self.torch.inference_mode():
            output = self.model.get_image_features(pixel_values=pixels)
        return _normalize(output.float().cpu().numpy())

    def _pixels(self, images):
        size = int(getattr(self.model.config.vision_config, "image_size", 224))
        mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype="float32")
        std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype="float32")
        arrays = np.stack([(np.asarray(_crop(image, size), dtype="float32") / 255.0 - mean) / std for image in images])
        return self.torch.from_numpy(arrays).permute(3, 0, 1, 2).unsqueeze(0)


def _crop(image, size: int):
    from PIL import Image

    width, height = image.size
    scale = size / min(width, height)
    resized = (max(size, round(width * scale)), max(size, round(height * scale)))
    image = image.resize(resized, Image.BICUBIC)
    left = (resized[0] - size) // 2
    top = (resized[1] - size) // 2
    return image.crop((left, top, left + size, top + size))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-9)).astype("float32")


def _response(vectors: np.ndarray) -> dict:
    return {
        "ok": True,
        "shape": list(vectors.shape),
        "data": base64.b64encode(vectors.tobytes()).decode("ascii"),
    }


def main() -> int:
    runtime = Runtime()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "ping":
                runtime.load()
                response = {
                    "ok": True,
                    "model": MODEL_NAME,
                    "dimension": runtime.dimension,
                    "frames": runtime.frame_count,
                    "device": runtime.device,
                }
            elif op == "texts":
                response = _response(runtime.texts(request.get("texts") or []))
            elif op == "images":
                response = _response(runtime.images(request.get("paths") or []))
            elif op == "quit":
                response = {"ok": True}
                print(json.dumps(response), flush=True)
                return 0
            else:
                response = {"ok": False, "error": f"Unknown operation: {op}"}
        except Exception as exc:  # noqa: BLE001
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-2000:],
            }
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
