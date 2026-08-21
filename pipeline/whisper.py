"""Persistent URL transcription for the GPU worker."""

from __future__ import annotations

import os
import threading
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .config import SETTINGS

try:
    from faster_whisper import WhisperModel
except ImportError as exc:  # pragma: no cover - reported through health()
    WhisperModel = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = os.getenv("WHISPER_DEVICE", SETTINGS.device)
COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "float16" if DEVICE == "cuda" else "int8",
)
MAX_DURATION = float(os.getenv("WHISPER_MAX_DURATION", "1800"))
MAX_BYTES = int(os.getenv("WHISPER_MAX_BYTES", str(1024 * 1024 * 1024)))

_MODEL: Any = None
_MODEL_ERROR = ""
_MODEL_LOCK = threading.Lock()
_TRANSCRIBE_LOCK = threading.Lock()


def _load_model() -> Any:
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        if _IMPORT_ERROR is not None:
            _MODEL_ERROR = "faster-whisper is not installed"
            raise RuntimeError(_MODEL_ERROR) from _IMPORT_ERROR
        try:
            _MODEL = WhisperModel(
                MODEL_NAME,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced by health/transcribe
            _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
            raise
        return _MODEL


def warm_model() -> None:
    """Load Whisper during worker startup so requests never load it lazily."""
    try:
        _load_model()
    except Exception:
        # Keep the worker API alive so /health exposes the actionable error.
        # A later request retries the load after configuration/dependency fixes.
        pass


def health() -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "loaded": _MODEL is not None,
        "error": _MODEL_ERROR,
    }


def transcribe_url(url: str, progress: Callable[[str, float], None] | None = None) -> dict[str, Any]:
    url = str(url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Please provide a complete http(s) URL.")

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed in the worker environment.") from exc

    with _TRANSCRIBE_LOCK:
        with tempfile.TemporaryDirectory(prefix="tiktokgen-whisper-") as raw_dir:
            folder = Path(raw_dir)
            options = {
                "format": "bestaudio/best",
                "outtmpl": str(folder / "source.%(ext)s"),
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "restrictfilenames": True,
            }
            if progress:
                progress("Downloading media on the GPU worker...", 0.08)
            try:
                with yt_dlp.YoutubeDL(options) as downloader:
                    info = downloader.extract_info(url, download=True)
                    prepared = Path(downloader.prepare_filename(info))
            except Exception as exc:  # noqa: BLE001 - extractor errors vary by platform
                raise RuntimeError(f"Could not download the URL: {exc}") from exc

            candidates = [prepared, *folder.iterdir()]
            media = next(
                (path for path in candidates if path.is_file() and path.stat().st_size),
                None,
            )
            if media is None:
                raise RuntimeError("The URL did not produce a downloadable media file.")
            if media.stat().st_size > MAX_BYTES:
                raise ValueError("The media file is too large for transcription.")
            duration = float(info.get("duration") or 0)
            if duration and duration > MAX_DURATION:
                raise ValueError(f"The video exceeds the {MAX_DURATION:.0f}-second limit.")

            if progress:
                progress("Transcribing with faster-whisper...", 0.42)
            segments, detected = _load_model().transcribe(
                str(media),
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            )
            output_segments = []
            text_parts = []
            for segment in segments:
                segment_text = (segment.text or "").strip()
                if segment_text:
                    text_parts.append(segment_text)
                words = [
                    {
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "text": word.word.strip(),
                    }
                    for word in (segment.words or [])
                    if word.word and word.start is not None and word.end is not None
                ]
                output_segments.append({
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": segment_text,
                    "words": words,
                })

    text = " ".join(text_parts).strip()
    if not text:
        raise ValueError("Whisper did not detect spoken text.")
    if progress:
        progress("Transcription finished", 1.0)
    return {
        "text": text,
        "language_code": getattr(detected, "language", "") or "",
        "duration": duration,
        "segments": output_segments,
        "source": url,
    }
