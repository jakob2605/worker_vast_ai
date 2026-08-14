"""
Worker API - runs ON the rented Vast.ai GPU instance.

Speaks HTTP on a mapped port, guarded by a shared token. No SSH needed for the
control path. The local dashboard is the only client.

Start:  WORKER_TOKEN=xxx uvicorn worker:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import db  # noqa: E402
from pipeline.config import (  # noqa: E402
    CLIPS_DIR,
    EMBEDDINGS_DIR,
    FRAMES_DIR,
    LIBRARY_DIR,
    METADATA_DIR,
    MOVIES_DIR,
    SETTINGS,
    ensure_library_dirs,
)
from pipeline.processor import (  # noqa: E402
    ingest_url,
    is_processing,
    pause_processing,
    record_source_link,
    reset_processing_outputs,
    start_semantics_only,
    start_processing,
)
from pipeline.semantics import SemanticAnalyzer  # noqa: E402
from pipeline.video_tools import ffprobe, file_sha256, has_nvenc, nvenc_usable  # noqa: E402

TOKEN = os.getenv("WORKER_TOKEN", "")
STARTED_AT = time.time()
SHUTDOWN_TIMER_PID = Path("/workspace/shutdown_timer.pid")
SHUTDOWN_TIMER_LOG = Path("/workspace/shutdown_timer.log")

app = FastAPI(title="Movie clips GPU worker")


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def auth(x_worker_token: str = Header(default="")) -> None:
    if not TOKEN:
        return  # unset means local testing
    if x_worker_token != TOKEN:
        raise HTTPException(401, "Bad or missing X-Worker-Token")


def clean_filename(name: str, fallback: str = "upload.mp4") -> str:
    cleaned = Path((name or "").replace("\\", "/")).name.strip()
    cleaned = "".join("_" if ch in '<>:"/\\|?*\x00' else ch for ch in cleaned).strip(" .")
    return cleaned or fallback


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        "nvenc_usable": nvenc_usable(),
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


@app.get("/usage", dependencies=[Depends(auth)])
def usage() -> dict[str, Any]:
    gpu_rows: list[dict[str, Any]] = []
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        for line in raw.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 8:
                gpu_rows.append({
                    "name": parts[0],
                    "gpu_util_pct": parse_float(parts[1]),
                    "memory_util_pct": parse_float(parts[2]),
                    "memory_used_mb": parse_float(parts[3]),
                    "memory_total_mb": parse_float(parts[4]),
                    "temperature_c": parse_float(parts[5]),
                    "power_draw_w": parse_float(parts[6]),
                    "power_limit_w": parse_float(parts[7]),
                })
    except Exception as exc:  # noqa: BLE001
        gpu_rows.append({"error": f"{type(exc).__name__}: {exc}"})

    disk = shutil.disk_usage(LIBRARY_DIR if LIBRARY_DIR.exists() else Path("/"))
    mem: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw_value = line.split(":", 1)
            value_kb = parse_float(raw_value.strip().split()[0])
            if value_kb is not None:
                mem[key] = round(value_kb / 1024, 1)
    except Exception:  # noqa: BLE001
        pass
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    return {
        "gpu": gpu_rows,
        "cpu_load": {"1m": load[0], "5m": load[1], "15m": load[2]},
        "memory": {
            "total_mb": mem.get("MemTotal"),
            "available_mb": mem.get("MemAvailable"),
            "used_mb": round((mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)), 1) if mem else None,
        },
        "disk": {
            "path": str(LIBRARY_DIR),
            "total_gb": round(disk.total / 1e9, 1),
            "used_gb": round(disk.used / 1e9, 1),
            "free_gb": round(disk.free / 1e9, 1),
        },
        "timestamp": time.time(),
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


class ShutdownReq(BaseModel):
    minutes: int


@app.post("/shutdown-later", dependencies=[Depends(auth)])
def shutdown_later(req: ShutdownReq) -> dict[str, Any]:
    """
    Schedule the instance/container to stop from inside the box.

    The timer is a detached shell process, so it continues even if the local
    dashboard or the worker process exits. This is intended as a cost guard;
    persistent disk charges may still apply depending on the Vast instance
    state after the container stops.
    """
    minutes = int(req.minutes)
    if minutes < 1 or minutes > 10080:
        raise HTTPException(400, "minutes must be between 1 and 10080")

    if SHUTDOWN_TIMER_PID.exists():
        try:
            old_pid = int(SHUTDOWN_TIMER_PID.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
            raise HTTPException(409, f"Shutdown timer already scheduled as PID {old_pid}")
        except ProcessLookupError:
            SHUTDOWN_TIMER_PID.unlink(missing_ok=True)
        except ValueError:
            SHUTDOWN_TIMER_PID.unlink(missing_ok=True)

    seconds = minutes * 60
    due_at = time.time() + seconds
    due_iso = datetime.fromtimestamp(due_at, timezone.utc).isoformat(timespec="seconds")
    script = f"""
set -eu
echo "scheduled stop for {due_iso} after {minutes} minute(s)" >> "{SHUTDOWN_TIMER_LOG}"
sleep {seconds}
echo "stopping instance/container at $(date -u)" >> "{SHUTDOWN_TIMER_LOG}"
sync || true
shutdown -h now >> "{SHUTDOWN_TIMER_LOG}" 2>&1 || true
poweroff >> "{SHUTDOWN_TIMER_LOG}" 2>&1 || true
kill -TERM 1 >> "{SHUTDOWN_TIMER_LOG}" 2>&1 || true
sleep 15
kill -KILL 1 >> "{SHUTDOWN_TIMER_LOG}" 2>&1 || true
"""
    proc = subprocess.Popen(
        ["bash", "-lc", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    SHUTDOWN_TIMER_PID.write_text(str(proc.pid), encoding="utf-8")
    return {
        "scheduled": True,
        "minutes": minutes,
        "seconds": seconds,
        "due_at": due_at,
        "due_at_utc": due_iso,
        "pid": proc.pid,
        "log": str(SHUTDOWN_TIMER_LOG),
        "note": "Timer runs on the Vast instance; your laptop does not need to stay on.",
    }


@app.post("/shutdown-later/cancel", dependencies=[Depends(auth)])
def cancel_shutdown_later() -> dict[str, Any]:
    """Cancel a pending shutdown timer that was scheduled by this worker."""
    if not SHUTDOWN_TIMER_PID.exists():
        return {"cancelled": False, "message": "No shutdown timer is scheduled."}

    try:
        pid = int(SHUTDOWN_TIMER_PID.read_text(encoding="utf-8").strip())
    except ValueError:
        SHUTDOWN_TIMER_PID.unlink(missing_ok=True)
        return {"cancelled": False, "message": "Removed invalid shutdown timer pid file."}

    try:
        os.killpg(pid, signal.SIGTERM)
        cancelled = True
        message = f"Cancelled shutdown timer PID {pid}."
    except ProcessLookupError:
        cancelled = False
        message = f"Shutdown timer PID {pid} was not running."
    finally:
        SHUTDOWN_TIMER_PID.unlink(missing_ok=True)

    try:
        with SHUTDOWN_TIMER_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{message} at {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    except OSError:
        pass
    return {"cancelled": cancelled, "pid": pid, "message": message, "log": str(SHUTDOWN_TIMER_LOG)}


@app.get("/gpu", dependencies=[Depends(auth)])
def gpu_detail() -> dict[str, Any]:
    analyzer = SemanticAnalyzer()
    analyzer._ensure_model()  # noqa: SLF001 - deliberate warm-up probe
    return {"semantics": analyzer.device_info(), "device": SETTINGS.device, "nvenc": has_nvenc()}


# --------------------------------------------------------------------------
class IngestReq(BaseModel):
    urls: list[str]
    movie_name: str
    autostart: bool = True
    allow_reprocess: bool = False


@app.post("/jobs", dependencies=[Depends(auth)])
def create_jobs(req: IngestReq) -> dict[str, Any]:
    """Queue one or more movie URLs. Downloads happen here, at datacenter speed."""
    accepted: list[dict[str, Any]] = []
    urls = [u.strip() for u in req.urls if u.strip()]
    movie_name = req.movie_name.strip()
    if not movie_name:
        raise HTTPException(400, "movie_name is required")
    if not urls:
        raise HTTPException(400, "At least one movie URL is required")
    for index, url in enumerate(urls, start=1):
        original_name = movie_name if len(urls) == 1 else f"{movie_name} {index}"
        holder: dict[str, Any] = {"url": url, "original_name": original_name}
        existing = db.find_movie_by_source_url(url)
        if existing:
            holder["duplicate_movie_id"] = existing["id"]
            holder["duplicate_of"] = existing["original_name"]
            if not req.allow_reprocess:
                holder["error"] = "Duplicate source URL. Enable reprocess existing movie to restart it."
                accepted.append(holder)
                continue
            if is_processing(int(existing["id"])):
                holder["error"] = "Existing movie is currently running. Pause or wait before reprocessing it."
                accepted.append(holder)
                continue
            cleanup = reset_processing_outputs(int(existing["id"]))
            holder["movie_id"] = existing["id"]
            holder["reprocessed_existing"] = True
            holder.update(cleanup)
            if req.autostart:
                start_processing(int(existing["id"]))
            accepted.append(holder)
            continue

        def run(u: str = url, name: str = original_name, h: dict[str, Any] = holder) -> None:
            try:
                movie_id = ingest_url(u, original_name=name, collection_title=movie_name)
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
        movie["clip_count"] = db.count_clips({"movie_id": movie["id"]})
        movie["downloadable_count"] = db.count_clips({"movie_id": movie["id"], "has_file": True})
        movie["motion_count"] = db.count_clips({"movie_id": movie["id"], "status": "motion_analyzed"})
        movie["indexed_count"] = db.count_clips({"movie_id": movie["id"], "status": "indexed"})
    return {"movies": movies}


@app.get("/titles", dependencies=[Depends(auth)])
def titles() -> dict[str, Any]:
    return {"titles": db.list_collection_titles()}


@app.post("/uploads", dependencies=[Depends(auth)])
async def upload_jobs(
    title: str = Form(...),
    autostart: bool = Form(True),
    allow_reprocess: bool = Form(False),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    collection_title = title.strip()
    if not collection_title:
        raise HTTPException(400, "title is required")
    if not files:
        raise HTTPException(400, "At least one file is required")

    accepted: list[dict[str, Any]] = []
    for file in files:
        original_name = clean_filename(file.filename)
        suffix = Path(original_name).suffix.lower()
        if suffix and suffix not in VIDEO_SUFFIXES:
            accepted.append({"original_name": original_name, "error": "Skipped non-video file"})
            continue

        filename = f"{int(time.time() * 1000)}_{len(accepted) + 1:04d}_{original_name}"
        target = MOVIES_DIR / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    handle.write(chunk)
            info = ffprobe(target)
            checksum = file_sha256(target)
            existing = db.find_movie_by_checksum(checksum)
            if existing:
                target.unlink(missing_ok=True)
                item = {
                    "original_name": original_name,
                    "duplicate_movie_id": existing["id"],
                    "duplicate_of": existing["original_name"],
                }
                if not allow_reprocess:
                    item["error"] = "Duplicate file. Enable reprocess existing movie to restart it."
                    accepted.append(item)
                    continue
                if is_processing(int(existing["id"])):
                    item["error"] = "Existing movie is currently running. Pause or wait before reprocessing it."
                    accepted.append(item)
                    continue
                cleanup = reset_processing_outputs(int(existing["id"]))
                db.update_movie(int(existing["id"]), collection_title=collection_title)
                if autostart:
                    start_processing(int(existing["id"]))
                accepted.append({
                    **item,
                    **cleanup,
                    "movie_id": existing["id"],
                    "reprocessed_existing": True,
                })
                continue
            movie_id = db.create_movie(
                original_name=original_name,
                filename=filename,
                path=target,
                checksum=checksum,
                duration=float(info["duration"]),
                fps=float(info["fps"]),
                width=int(info["width"]),
                height=int(info["height"]),
                collection_title=collection_title,
            )
            source_url = f"local-upload:{original_name}"
            db.update_movie(
                movie_id,
                status="imported",
                progress_stage="imported",
                progress_detail="Ready to process",
                source_url=source_url,
            )
            record_source_link(movie_id, original_name, collection_title, source_url, target, source_type="upload")
            if autostart:
                start_processing(movie_id)
            accepted.append({"movie_id": movie_id, "original_name": original_name})
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            accepted.append({"original_name": original_name, "error": str(exc)})
        finally:
            await file.close()
    created = sum(1 for item in accepted if item.get("movie_id"))
    return {"accepted": created, "items": accepted, "collection_title": collection_title}


@app.post("/jobs/{movie_id}/start", dependencies=[Depends(auth)])
def start_job(movie_id: int) -> dict[str, Any]:
    movie = db.get_movie(movie_id)
    if not movie:
        raise HTTPException(404, "Movie not found")
    if is_processing(movie_id):
        return {"started": False, "movie": movie, "message": "Movie is already running."}
    cleanup: dict[str, int] | None = None
    if not movie.get("paused") and db.count_clips({"movie_id": movie_id}) > 0:
        cleanup = reset_processing_outputs(movie_id)
    return {"started": start_processing(movie_id), "movie": db.get_movie(movie_id), "cleanup": cleanup}


@app.post("/jobs/{movie_id}/pause", dependencies=[Depends(auth)])
def pause_job(movie_id: int) -> dict[str, Any]:
    if not db.get_movie(movie_id):
        raise HTTPException(404, "Movie not found")
    pause_processing(movie_id)
    return {"movie": db.get_movie(movie_id)}


@app.post("/jobs/{movie_id}/semantics", dependencies=[Depends(auth)])
def rerun_semantics_job(movie_id: int) -> dict[str, Any]:
    movie = db.get_movie(movie_id)
    if not movie:
        raise HTTPException(404, "Movie not found")
    if is_processing(movie_id):
        return {"started": False, "movie": movie, "message": "Movie is already running."}
    if db.count_clips({"movie_id": movie_id}) == 0:
        raise HTTPException(409, "No clips exist yet. Run the full job first.")
    return {"started": start_semantics_only(movie_id), "movie": db.get_movie(movie_id)}


@app.delete("/jobs/{movie_id}", dependencies=[Depends(auth)])
def delete_job(movie_id: int) -> dict[str, Any]:
    movie = db.get_movie(movie_id)
    if not movie:
        raise HTTPException(404, "Movie not found")
    active = {
        "queued",
        "downloading",
        "detecting_shots",
        "exporting_clips",
        "motion_analysis",
        "semantic_indexing",
        "metadata_export",
    }
    if movie.get("progress_stage") in active:
        raise HTTPException(409, "Pause the job before deleting it.")

    cleanup = reset_processing_outputs(movie_id, delete_source=True)
    db.delete_movie(movie_id)
    return {"deleted": True, **cleanup}


# --------------------------------------------------------------------------
@app.get("/clips", dependencies=[Depends(auth)])
def clips(
    movie_id: Optional[int] = None,
    collection_title: Optional[str] = None,
    text: Optional[str] = None,
    shot_size: Optional[str] = None,
    camera_motion_type: Optional[str] = None,
    animation_motion_bucket: Optional[str] = None,
    mood: Optional[str] = None,
    people_count: Optional[str] = None,
    tag: Optional[str] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    has_file: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
) -> dict[str, Any]:
    filters = {
        "movie_id": movie_id,
        "collection_title": collection_title,
        "text": text,
        "shot_size": shot_size,
        "camera_motion_type": camera_motion_type,
        "animation_motion_bucket": animation_motion_bucket,
        "mood": mood,
        "people_count": people_count,
        "tag": tag,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "has_file": has_file,
    }
    rows = db.list_clips(filters, limit=limit, offset=offset)
    for row in rows:
        path = row.get("clip_path")
        row["size_mb"] = round(Path(path).stat().st_size / 1048576, 2) if path and Path(path).exists() else None
    return {
        "clips": rows,
        "count": db.count_clips(filters),
        "downloadable_count": db.count_clips({**filters, "has_file": True}),
        "limit": limit,
        "offset": offset,
    }


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
