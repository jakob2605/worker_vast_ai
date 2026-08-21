from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from http.cookiejar import CookieJar
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urljoin, urlparse
import urllib.request


CHUNK_SIZE = 4 * 1024 * 1024
ProgressCallback = Callable[[int, int], None]


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


def file_fingerprint(
    path: Path,
    metadata: dict[str, float | int] | None = None,
    sample_size: int = 1024 * 1024,
) -> str:
    """Create a fast duplicate fingerprint without reading the whole movie.

    The file size, ffprobe metadata, and up to 1 MiB from the beginning, middle,
    and end are hashed. This is not a cryptographic replacement for SHA-256 of
    the complete file, but is substantially faster for large video uploads and
    much safer than comparing file size alone.
    """
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(f"size:{size};metadata:".encode("utf-8"))
    digest.update(json.dumps(metadata or {}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b";samples:")

    offsets = sorted({
        0,
        max(0, size // 2 - sample_size // 2),
        max(0, size - sample_size),
    })
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(handle.read(sample_size))
    return f"sampled-sha256:{digest.hexdigest()}"


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


def convert_gif_to_mp4(source: Path, target: Path) -> Path:
    """Convert one uploaded GIF to a seekable MP4 movie."""
    require_command("ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        # Ignore an infinite GIF loop and import the animation once.
        "-ignore_loop", "1", "-i", str(source),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ]
    subprocess.check_call(command)
    if not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"GIF conversion produced no MP4: {source}")
    return target


def has_nvenc() -> bool:
    """True when this ffmpeg build exposes the NVIDIA h264 encoder."""
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT
        )
        return "h264_nvenc" in out
    except Exception:  # noqa: BLE001
        return False


def nvenc_usable() -> bool:
    """True when this container can actually open an NVENC encode session."""
    try:
        subprocess.check_call(
            [
                "ffmpeg", "-hide_banner",
                "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
                "-t", "1",
                "-c:v", "h264_nvenc",
                "-gpu", "0",
                "-preset", "p5",
                "-rc", "vbr",
                "-cq", "21",
                "-f", "null", "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
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

    if use_nvenc and nvenc_usable():
        try:
            subprocess.check_call(build("h264_nvenc"))
            return "h264_nvenc"
        except subprocess.CalledProcessError:
            target.unlink(missing_ok=True)  # partial file from the failed attempt
    subprocess.check_call(build("libx264"))
    return "libx264"


class _GoogleDriveConfirmParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[tuple[str, dict[str, str]]] = []
        self.links: list[str] = []
        self._form_action: str | None = None
        self._form_inputs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag == "form":
            self._form_action = values.get("action", "")
            self._form_inputs = {}
        elif tag == "input" and self._form_action is not None:
            name = values.get("name")
            if name:
                self._form_inputs[name] = values.get("value", "")
        elif tag == "a":
            href = values.get("href", "")
            if href and "confirm=" in href and "download" in href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._form_action is not None:
            self.forms.append((self._form_action, dict(self._form_inputs)))
            self._form_action = None
            self._form_inputs = {}


def _request(url: str, user_agent: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": user_agent})


def _is_google_drive_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host.endswith("drive.google.com") or host.endswith("drive.usercontent.google.com")


def _google_drive_confirm_url(base_url: str, html: str) -> str | None:
    parser = _GoogleDriveConfirmParser()
    parser.feed(html)

    for action, fields in parser.forms:
        if "confirm" not in fields:
            continue
        if "download" not in action and "drive.usercontent.google.com" not in action:
            continue
        return urljoin(base_url, action) + "?" + urlencode(fields)

    if parser.links:
        return urljoin(base_url, parser.links[0])
    return None


def _stream_response(response, target: Path, progress: ProgressCallback | None = None) -> None:
    total = int(response.headers.get("Content-Length") or 0)
    done = 0
    with target.open("wb") as handle:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)


def download_movie(url: str, target: Path, progress: ProgressCallback | None = None) -> Path:
    """
    Stream a movie straight onto this machine. This is the whole point of running
    on a rented box: hundreds of GB arrive at datacenter bandwidth and never
    travel through the user's home connection.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    user_agent = "movie-clips-worker/1.0"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    with opener.open(_request(url, user_agent), timeout=60) as response:  # noqa: S310
        content_type = response.headers.get("Content-Type", "").lower()
        if _is_google_drive_url(response.url) and "text/html" in content_type:
            html = response.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
            confirm_url = _google_drive_confirm_url(response.url, html)
            if not confirm_url:
                raise RuntimeError("Google Drive returned a confirmation page, but no download confirmation link was found.")
            with opener.open(_request(confirm_url, user_agent), timeout=60) as confirmed:  # noqa: S310
                if "text/html" in confirmed.headers.get("Content-Type", "").lower():
                    raise RuntimeError("Google Drive did not return the movie file after confirmation.")
                _stream_response(confirmed, target, progress)
        else:
            _stream_response(response, target, progress)

    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded 0 bytes from {url}")
    return target
