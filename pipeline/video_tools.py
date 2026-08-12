from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


class ToolMissingError(RuntimeError):
    pass


def require_command(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise ToolMissingError(f"{name} is required but was not found on PATH.")
    return resolved


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, float | int]:
    require_command("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    def fps_from_rate(rate: str) -> float:
        if not rate or "/" not in rate:
            return 0.0
        num, den = rate.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0

    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    return {
        "duration": duration,
        "fps": fps_from_rate(stream.get("r_frame_rate", "")),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def has_nvenc() -> bool:
    """True when this ffmpeg build exposes the NVIDIA h264 encoder."""
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT
        )
        return "h264_nvenc" in out
    except Exception:  # noqa: BLE001
        return False


def export_clip(
    source: Path,
    target: Path,
    start_time: float,
    end_time: float,
    crf: int,
    preset: str,
    use_nvenc: bool = False,
    nvenc_preset: str = "p5",
    nvenc_cq: int = 21,
) -> str:
    """
    Cut one clip. Returns the encoder actually used.

    NVENC moves encoding onto the GPU - the third and largest win, since
    libx264 at preset 'slow' was the dominant cost of the original pipeline.
    Falls back to libx264 automatically if NVENC is unavailable or fails.
    """
    require_command("ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end_time - start_time)

    def build(encoder: str) -> list[str]:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            # -ss before -i seeks fast; keep the original's frame-accurate behaviour
            # by re-decoding from the nearest keyframe.
            "-ss", f"{start_time:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-map", "0:v:0", "-map", "0:a?",
        ]
        if encoder == "h264_nvenc":
            cmd += ["-c:v", "h264_nvenc", "-preset", nvenc_preset, "-rc", "vbr", "-cq", str(nvenc_cq)]
        else:
            cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
        cmd += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(target)]
        return cmd

    if use_nvenc:
        try:
            subprocess.check_call(build("h264_nvenc"))
            return "h264_nvenc"
        except subprocess.CalledProcessError:
            target.unlink(missing_ok=True)  # partial file from the failed attempt
    subprocess.check_call(build("libx264"))
    return "libx264"


def download_movie(url: str, target: Path, progress: "callable | None" = None) -> Path:
    """
    Stream a movie straight onto this machine. This is the whole point of running
    on a rented box: hundreds of GB arrive at datacenter bandwidth and never
    travel through the user's home connection.
    """
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "movie-clips-worker/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        chunk_size = 4 * 1024 * 1024
        with target.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded 0 bytes from {url}")
    return target

