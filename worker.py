"""
Worker API - runs ON the rented Vast.ai GPU instance.

Speaks HTTP on a mapped port, guarded by a shared token. No SSH needed for the
control path. The local dashboard is the only client.

Start:  WORKER_TOKEN=xxx uvicorn worker:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import db  # noqa: E402
from pipeline.config import (  # noqa: E402
    CLIPS_DIR,
    EMBEDDINGS_DIR,
    LIBRARY_DIR,
    METADATA_DIR,
    MOVIES_DIR,
    SETTINGS,
    ensure_library_dirs,
)
from pipeline.processor import ingest_url, pause_processing, start_processing  # noqa: E402
from pipeline.semantics import SemanticAnalyzer  # noqa: E402
from pipeline.video_tools import has_nvenc  # noqa: E402

TOKEN = os.getenv("WORKER_TOKEN", "")
STARTED_AT = time.time()

app = FastAPI(title="Movie clips GPU worker")


def auth(x_worker_token: str = Header(default="")) -> None:
    if not TOKEN:
        return  # unset means local testing
    if x_worker_token != TOKEN:
        raise HTTPException(401, "Bad or missing X-Worker-Token")


@app.on_event("startup")
def startup() -> None:
    ensure_library_dirs()
    db.init_db()


# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    """Unauthenticated so the dashboard can tell 'not up yet' from 'wrong token'."""
    gpu: dict[str, Any] = {"name": None, "memory_total_mb": None, "driver": None}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().splitlines()
        if out:
            name, mem, driver = (part.strip() for part in out[0].split(","))
            gpu = {"name": name, "memory_total_mb": int(float(mem)), "driver": driver}
    except Exception:  # noqa: BLE001
        pass

    usage = shutil.disk_usage(LIBRARY_DIR if LIBRARY_DIR.exists() else Path("/"))
    return {
        "ok": True,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "device": SETTINGS.device,
        "gpu": gpu,
        "nvenc": has_nvenc(),
        "use_nvenc": SETTINGS.use_nvenc,
        "siglip_batch": SETTINGS.siglip_batch_size,
        "fp16": SETTINGS.siglip_fp16,
        "library": str(LIBRARY_DIR),
        "disk": {
            "total_gb": round(usage.total / 1e9, 1),
            "used_gb": round(usage.used / 1e9, 1),
            "free_gb": round(usage.free / 1e9, 1),
        },
        "auth_required": bool(TOKEN),
    }


@app.post("/update", dependencies=[Depends(auth)])
def update() -> dict[str, Any]:
    """
    git pull, then restart this process so the new code takes effect.

    Saves a stop/start cycle every time the pipeline changes. Settings live in
    config.py and are read at import, so the re-exec picks them up.
    """
    worker_dir = Path(__file__).resolve().parent
    result: dict[str, Any] = {"dir": str(worker_dir)}

    pull = subprocess.run(
        ["git", "-C", str(worker_dir), "pull", "--ff-only"],
        capture_output=True, text=True, timeout=120,
    )
    result["pull_stdout"] = pull.stdout.strip()
    result["pull_stderr"] = pull.stderr.strip()
    result["pull_ok"] = pull.returncode == 0
    if not result["pull_ok"]:
        raise HTTPException(502, f"git pull failed: {pull.stderr.strip() or pull.stdout.strip()}")

    result["head"] = subprocess.run(
        ["git", "-C", str(worker_dir), "log", "-1", "--oneline"],
        capture_output=True, text=True,
    ).stdout.strip()

    port = os.getenv("WORKER_PORT", "8100")

    def restart() -> None:
        time.sleep(0.5)  # let this response flush first
        os.chdir(worker_dir)
        os.execv(
            sys.executable,
            [sys.executable, "-m", "uvicorn", "worker:app", "--host", "0.0.0.0", "--port", port],
        )

    threading.Thread(target=restart, daemon=True).start()
    result["restarting"] = True
    return result


@app.get("/gpu", dependencies=[Depends(auth)])
def gpu_detail() -> dict[str, Any]:
    analyzer = SemanticAnalyzer()
    analyzer._ensure_model()  # noqa: SLF001 - deliberate warm-up probe
    return {"semantics": analyzer.device_info(), "device": SETTINGS.device, "nvenc": has_nvenc()}


# --------------------------------------------------------------------------
class IngestReq(BaseModel):
    urls: list[str]
    autostart: bool = True


@app.post("/jobs", dependencies=[Depends(auth)])
def create_jobs(req: IngestReq) -> dict[str, Any]:
    """Queue one or more movie URLs. Downloads happen here, at datacenter speed."""
    accepted: list[dict[str, Any]] = []
    for url in [u.strip() for u in req.urls if u.strip()]:
        holder: dict[str, Any] = {"url": url}

        def run(u: str = url, h: dict[str, Any] = holder) -> None:
            try:
                movie_id = ingest_url(u)
                h["movie_id"] = movie_id
                if req.autostart:
                    start_processing(movie_id)
            except Exception as exc:  # noqa: BLE001
                h["error"] = str(exc)

        threading.Thread(target=run, daemon=True).start()
        accepted.append(holder)
    return {"accepted": len(accepted), "jobs": accepted}


@app.get("/jobs", dependencies=[Depends(auth)])
def jobs() -> dict[str, Any]:
    movies = db.list_movies()
    for movie in movies:
        clips = db.list_clips({"movie_id": movie["id"]})
        movie["clip_count"] = len(clips)
        movie["indexed_count"] = sum(1 for c in clips if c.get("status") == "indexed")
    return {"movies": movies}


@app.post("/jobs/{movie_id}/start", dependencies=[Depends(auth)])
def start_job(movie_id: int) -> dict[str, Any]:
    if not db.get_movie(movie_id):
        raise HTTPException(404, "Movie not found")
    return {"started": start_processing(movie_id), "movie": db.get_movie(movie_id)}


@app.post("/jobs/{movie_id}/pause", dependencies=[Depends(auth)])
def pause_job(movie_id: int) -> dict[str, Any]:
    if not db.get_movie(movie_id):
        raise HTTPException(404, "Movie not found")
    pause_processing(movie_id)
    return {"movie": db.get_movie(movie_id)}


# --------------------------------------------------------------------------
@app.get("/clips", dependencies=[Depends(auth)])
def clips(
    movie_id: Optional[int] = None,
    text: Optional[str] = None,
    shot_size: Optional[str] = None,
    camera_motion_type: Optional[str] = None,
    mood: Optional[str] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
) -> dict[str, Any]:
    rows = db.list_clips(
        {
            "movie_id": movie_id, "text": text, "shot_size": shot_size,
            "camera_motion_type": camera_motion_type, "mood": mood,
            "min_duration": min_duration, "max_duration": max_duration,
        }
    )
    for row in rows:
        path = row.get("clip_path")
        row["size_mb"] = round(Path(path).stat().st_size / 1048576, 2) if path and Path(path).exists() else None
    return {"clips": rows, "count": len(rows)}


@app.get("/clips/{clip_id}/file", dependencies=[Depends(auth)])
def clip_file(clip_id: int) -> FileResponse:
    """Pull one clip back. Nothing is transferred unless it is asked for."""
    clip = db.get_clip(clip_id)
    if not clip or not clip.get("clip_path"):
        raise HTTPException(404, "Clip not found")
    path = Path(clip["clip_path"])
    if not path.exists():
        raise HTTPException(404, "Clip file missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


class BundleReq(BaseModel):
    movie_id: Optional[int] = None
    include_frames: bool = False


@app.post("/bundle", dependencies=[Depends(auth)])
def bundle(req: BundleReq) -> FileResponse:
    """
    Zip the small artefacts - sqlite, metadata sidecars, embeddings - so the
    local library can mirror the remote one without moving any video.
    """
    out = LIBRARY_DIR / "bundle.zip"
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        db_path = LIBRARY_DIR / "movie_clips.sqlite3"
        if db_path.exists():
            zf.write(db_path, "movie_clips.sqlite3")
        for folder, label in ((METADATA_DIR, "metadata"), (EMBEDDINGS_DIR, "embeddings")):
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, f"{label}/{path.relative_to(folder)}")
        if req.include_frames:
            from pipeline.config import FRAMES_DIR

            for path in FRAMES_DIR.rglob("*.jpg"):
                zf.write(path, f"frames/{path.relative_to(FRAMES_DIR)}")
    return FileResponse(out, media_type="application/zip", filename="library_bundle.zip")


@app.get("/storage", dependencies=[Depends(auth)])
def storage() -> dict[str, Any]:
    def size_of(path: Path) -> float:
        if not path.exists():
            return 0.0
        return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9, 3)

    return {
        "movies_gb": size_of(MOVIES_DIR),
        "clips_gb": size_of(CLIPS_DIR),
        "frames_gb": size_of(LIBRARY_DIR / "frames"),
        "embeddings_gb": size_of(EMBEDDINGS_DIR),
    }


class PurgeReq(BaseModel):
    movies: bool = True


@app.post("/purge", dependencies=[Depends(auth)])
def purge(req: PurgeReq) -> dict[str, Any]:
    """Delete source movies once clips exist - they are the bulk of the disk."""
    freed = 0.0
    if req.movies:
        for path in MOVIES_DIR.glob("*"):
            if path.is_file():
                freed += path.stat().st_size
                path.unlink()
    return {"freed_gb": round(freed / 1e9, 3)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("WORKER_PORT", "8100")))
