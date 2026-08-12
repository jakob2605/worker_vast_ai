from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


import os

# On a Vast instance the library lives under /workspace so it can sit on a
# persistent volume; locally it falls back to a folder next to the worker.
ROOT = Path(os.getenv("WORKER_ROOT", Path(__file__).resolve().parent.parent)).resolve()
LIBRARY_DIR = Path(os.getenv("LIBRARY_DIR", ROOT / "library")).resolve()
MOVIES_DIR = LIBRARY_DIR / "movies"
CLIPS_DIR = LIBRARY_DIR / "clips"
METADATA_DIR = LIBRARY_DIR / "metadata"
FRAMES_DIR = LIBRARY_DIR / "frames"
EMBEDDINGS_DIR = LIBRARY_DIR / "embeddings"
DB_PATH = LIBRARY_DIR / "movie_clips.sqlite3"
STATIC_DIR = ROOT / "web"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def detect_device() -> str:
    """cuda when a GPU is actually usable, else cpu. Never raises."""
    if os.getenv("FORCE_DEVICE"):
        return os.environ["FORCE_DEVICE"]
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


DEVICE = detect_device()


@dataclass(frozen=True)
class ProcessingSettings:
    min_clip_seconds: float = 2.0
    merge_tiny_shots_seconds: float = 0.8
    transnet_threshold: float = 0.5
    sample_frames_per_clip: int = 5
    motion_resize_width: int = 480
    export_crf: int = 18
    export_preset: str = "slow"
    siglip_model: str = "google/siglip2-base-patch16-224"
    # --- GPU additions ---
    device: str = DEVICE
    # NVENC encodes clips on the GPU instead of libx264 on the CPU. Export was the
    # single biggest CPU cost in the original pipeline.
    use_nvenc: bool = _env_flag("USE_NVENC", DEVICE == "cuda")
    nvenc_preset: str = os.getenv("NVENC_PRESET", "p5")
    nvenc_cq: int = int(os.getenv("NVENC_CQ", "21"))
    # How many frames to push through SigLIP at once. Batching is where most of
    # the GPU speedup comes from; on CPU keep it small.
    siglip_batch_size: int = int(os.getenv("SIGLIP_BATCH", "32" if DEVICE == "cuda" else "5"))
    siglip_fp16: bool = _env_flag("SIGLIP_FP16", DEVICE == "cuda")
    export_workers: int = int(os.getenv("EXPORT_WORKERS", "4"))


SETTINGS = ProcessingSettings()


def ensure_library_dirs() -> None:
    for path in [LIBRARY_DIR, MOVIES_DIR, CLIPS_DIR, METADATA_DIR, FRAMES_DIR, EMBEDDINGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

