"""
Worker API - runs ON the rented Vast.ai GPU instance.

Speaks HTTP on a mapped port, guarded by a shared token. No SSH needed for the
control path. The local dashboard is the only client.

Start:  WORKER_TOKEN=xxx uvicorn worker:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import os
import json
import signal
import shutil
import subprocess
import sys
import threading
import time
import zipfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

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
from pipeline.filter_query import FilterQueryError  # noqa: E402
from pipeline.cloud_backup import (  # noqa: E402
    create_snapshot,
    list_snapshots,
    rclone_ready,
    restore_snapshot,
)
from pipeline.processor import (  # noqa: E402
    embed_text_for_profile,
    embed_texts_for_profile,
    ingest_url,
    is_processing,
    pause_processing,
    record_source_link,
    reset_processing_outputs,
    running_movie_ids,
    start_semantics_only,
    start_processing,
)
from pipeline.profiles import BUILTIN_PROFILES, DEFAULT_PROFILE_ID, get_profile  # noqa: E402
from pipeline.semantics import SemanticAnalyzer  # noqa: E402
from pipeline.video_tools import ffprobe, file_sha256, has_nvenc, nvenc_usable  # noqa: E402

TOKEN = os.getenv("WORKER_TOKEN", "")
STARTED_AT = time.time()
SHUTDOWN_TIMER_PID = Path("/workspace/shutdown_timer.pid")
SHUTDOWN_TIMER_LOG = Path("/workspace/shutdown_timer.log")

app = FastAPI(title="Movie clips GPU worker")
_CLOUD_JOBS: dict[str, dict[str, Any]] = {}
_CLOUD_JOBS_LOCK = threading.Lock()
_EMBEDDING_CACHE: dict[str, Any] = {"profile_id": "", "signature": (), "vectors": {}}
_EMBEDDING_CACHE_LOCK = threading.Lock()


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


def start_cloud_job(kind: str, operation, *args: Any, **kwargs: Any) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _CLOUD_JOBS_LOCK:
        _CLOUD_JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "message": "queued",
            "progress": 0.0,
            "result": None,
            "error": "",
        }

    def progress(message: str, value: float) -> None:
        with _CLOUD_JOBS_LOCK:
            _CLOUD_JOBS[job_id].update(message=message, progress=max(0.0, min(1.0, value)))

    def run() -> None:
        with _CLOUD_JOBS_LOCK:
            _CLOUD_JOBS[job_id]["status"] = "running"
        try:
            result = operation(*args, progress=progress, **kwargs)
            with _CLOUD_JOBS_LOCK:
                _CLOUD_JOBS[job_id].update(status="complete", progress=1.0, result=result)
        except Exception as exc:  # noqa: BLE001
            with _CLOUD_JOBS_LOCK:
                _CLOUD_JOBS[job_id].update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    message="failed",
                )

    threading.Thread(target=run, daemon=True).start()
    return job_id


def rank_semantic_clips(
    query: str,
    rows: list[dict[str, Any]],
    profile_id: str,
    matching_mode: str,
) -> list[dict[str, Any]]:
    import numpy as np

    get_profile(profile_id)
    if matching_mode not in {"mean_only", "mean_and_frames"}:
        raise HTTPException(400, "embedding_mode must be mean_only or mean_and_frames")
    query_embedding = embed_text_for_profile(profile_id, query)
    query_norm = float(np.linalg.norm(query_embedding))
    if query_norm <= 0:
        raise HTTPException(500, "Semantic query embedding has zero norm")
    query_embedding = query_embedding / query_norm

    ranked: list[dict[str, Any]] = []
    for row in rows:
        record = db.get_clip_embedding(int(row["id"]), profile_id)
        embedding_path = record.get("artifact_path") if record and record.get("status") == "complete" else None
        if not embedding_path and profile_id == DEFAULT_PROFILE_ID:
            embedding_path = row.get("embedding_path")
        if not embedding_path:
            continue
        path = Path(embedding_path)
        if path.suffix.lower() not in {".npy", ".npz"} or not path.exists():
            continue
        try:
            loaded = np.load(path)
            if path.suffix.lower() == ".npz":
                embedding = loaded["mean"].astype("float32")
                frames = loaded["frames"].astype("float32") if "frames" in loaded.files else np.zeros((0, 0), dtype="float32")
            else:
                embedding = loaded.astype("float32")
                frames = np.zeros((0, 0), dtype="float32")
        except Exception:
            continue
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            continue
        scored = dict(row)
        mean_score = float(np.dot(query_embedding, embedding / norm))
        frame_score = mean_score
        if matching_mode == "mean_and_frames" and frames.size:
            frame_norms = np.linalg.norm(frames, axis=1, keepdims=True)
            normalized_frames = frames / np.maximum(frame_norms, 1e-9)
            frame_score = max(mean_score, float(np.max(normalized_frames @ query_embedding)))
        scored["semantic_score"] = round(
            mean_score if matching_mode == "mean_only" else 0.6 * mean_score + 0.4 * frame_score,
            4,
        )
        scored["embedding_profile"] = profile_id
        scored["embedding_mode"] = matching_mode
        ranked.append(scored)
    ranked.sort(key=lambda item: item["semantic_score"], reverse=True)
    return ranked


class SemanticMatchReq(BaseModel):
    profile_id: str = DEFAULT_PROFILE_ID
    queries: list[str]
    anchor: str = ""
    anchor_weight: float = 0.3
    embedding_mode: str = "mean_and_frames"
    frame_weight: float = 0.4
    movie_id: Optional[int] = None
    collection_title: Optional[str] = None
    filter_query: Optional[str] = None
    limit: int = 80


def _profile_embedding_vectors(profile_id: str) -> dict[int, tuple[Any, Any, Any]]:
    import numpy as np

    records = [row for row in db.list_clip_embeddings(profile_id) if row.get("status") == "complete"]
    signature = tuple(
        (int(row["clip_id"]), str(row.get("updated_at") or ""), str(row.get("artifact_path") or ""))
        for row in records
    )
    with _EMBEDDING_CACHE_LOCK:
        if _EMBEDDING_CACHE["profile_id"] == profile_id and _EMBEDDING_CACHE["signature"] == signature:
            return _EMBEDDING_CACHE["vectors"]

        vectors: dict[int, tuple[Any, Any, Any]] = {}
        for record in records:
            path = Path(record.get("artifact_path") or "")
            if path.suffix.lower() not in {".npy", ".npz"} or not path.is_file():
                continue
            try:
                loaded = np.load(path)
                try:
                    if hasattr(loaded, "files"):
                        mean = np.asarray(loaded["mean"], dtype="float32").reshape(-1)
                        frames = (
                            np.asarray(loaded["frames"], dtype="float32")
                            if "frames" in loaded.files else np.zeros((0, mean.size), dtype="float32")
                        )
                        times = (
                            np.asarray(loaded["frame_times"], dtype="float32").reshape(-1)
                            if "frame_times" in loaded.files else np.zeros((0,), dtype="float32")
                        )
                    else:
                        mean = np.asarray(loaded, dtype="float32").reshape(-1)
                        frames = np.zeros((0, mean.size), dtype="float32")
                        times = np.zeros((0,), dtype="float32")
                finally:
                    if hasattr(loaded, "close"):
                        loaded.close()
            except Exception:
                continue
            mean /= max(float(np.linalg.norm(mean)), 1e-9)
            if frames.size:
                frames /= np.maximum(np.linalg.norm(frames, axis=1, keepdims=True), 1e-9)
            vectors[int(record["clip_id"])] = (mean, frames, times)
        _EMBEDDING_CACHE.update(profile_id=profile_id, signature=signature, vectors=vectors)
        return vectors


def _semantic_match(req: SemanticMatchReq) -> list[dict[str, Any]]:
    import numpy as np

    get_profile(req.profile_id)
    if req.embedding_mode not in {"mean_only", "mean_and_frames"}:
        raise HTTPException(400, "embedding_mode must be mean_only or mean_and_frames")
    queries = [text.strip() for text in req.queries if text and text.strip()]
    if not queries:
        raise HTTPException(400, "queries must contain at least one non-empty string")
    anchor = req.anchor.strip()
    texts = queries + ([anchor] if anchor else [])
    vectors = np.asarray(embed_texts_for_profile(req.profile_id, texts), dtype="float32")
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    filters = {
        "movie_id": req.movie_id,
        "collection_title": req.collection_title,
        "filter_query": req.filter_query,
        "has_file": True,
    }
    available_vectors = _profile_embedding_vectors(req.profile_id)
    candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    for row in db.list_clips(filters):
        cached = available_vectors.get(int(row["id"]))
        if cached is None:
            continue
        mean, frames, times = cached
        if mean.size != vectors.shape[1]:
            continue
        candidates.append((row, mean, frames, times))

    if not candidates:
        return []
    scores = np.zeros((len(queries), len(candidates)), dtype="float32")
    best_times = np.zeros((len(queries), len(candidates)), dtype="float32")
    frame_weight = 0.0 if req.embedding_mode == "mean_only" else min(1.0, max(0.0, req.frame_weight))
    for qi, query in enumerate(vectors[:len(queries)]):
        for ci, (_, mean, frames, times) in enumerate(candidates):
            mean_score = float(mean @ query)
            frame_score = mean_score
            if frame_weight and frames.size:
                frame_scores = frames @ query
                best_index = int(np.argmax(frame_scores))
                frame_score = max(mean_score, float(frame_scores[best_index]))
                if best_index < times.size:
                    best_times[qi, ci] = float(times[best_index])
            scores[qi, ci] = (1.0 - frame_weight) * mean_score + frame_weight * frame_score
    means = scores.mean(axis=1, keepdims=True)
    stds = scores.std(axis=1, keepdims=True)
    z_scores = (scores - means) / np.maximum(stds, 1e-6)
    winning_query = z_scores.argmax(axis=0)
    combined = z_scores.max(axis=0)
    if anchor:
        anchor_vector = vectors[-1]
        anchor_scores = np.zeros(len(candidates), dtype="float32")
        for ci, (_, mean, frames, _) in enumerate(candidates):
            mean_score = float(mean @ anchor_vector)
            frame_score = max(mean_score, float(np.max(frames @ anchor_vector))) if frame_weight and frames.size else mean_score
            anchor_scores[ci] = (1.0 - frame_weight) * mean_score + frame_weight * frame_score
        anchor_z = (anchor_scores - anchor_scores.mean()) / max(float(anchor_scores.std()), 1e-6)
        weight = min(1.0, max(0.0, req.anchor_weight))
        combined = (1.0 - weight) * combined + weight * anchor_z

    ranked: list[dict[str, Any]] = []
    for ci in np.argsort(-combined):
        row = dict(candidates[int(ci)][0])
        qi = int(winning_query[int(ci)])
        row.update({
            "z": round(float(combined[int(ci)]), 3),
            "z_best_query": round(float(z_scores[qi, int(ci)]), 3),
            "query": queries[qi],
            "query_index": qi,
            "best_frame_time": round(float(best_times[qi, int(ci)]), 4),
            "embedding_profile": req.profile_id,
            "embedding_mode": req.embedding_mode,
        })
        ranked.append(row)
        if len(ranked) >= max(1, min(int(req.limit), 1000)):
            break
    return ranked


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
        "embedding_profiles": [profile.to_dict() for profile in BUILTIN_PROFILES],
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


class TitleReq(BaseModel):
    title: str
    update_original_name: bool = False


class SemanticsReq(BaseModel):
    profile_id: str = DEFAULT_PROFILE_ID
    embeddings_per_clip: int | None = None
    overwrite: bool = False


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
        movie["embedding_profiles"] = db.list_embedding_profiles(int(movie["id"]))
    return {"movies": movies}


@app.get("/embedding-profiles", dependencies=[Depends(auth)])
def embedding_profiles(movie_id: Optional[int] = None) -> dict[str, Any]:
    if movie_id is not None and not db.get_movie(movie_id):
        raise HTTPException(404, "Movie not found")
    return {
        "profiles": db.list_embedding_profiles(movie_id),
        "default_profile_id": DEFAULT_PROFILE_ID,
        "matching_modes": ["mean_only", "mean_and_frames"],
    }


@app.get("/titles", dependencies=[Depends(auth)])
def titles() -> dict[str, Any]:
    return {"titles": db.list_collection_titles()}


@app.post("/jobs/{movie_id}/title", dependencies=[Depends(auth)])
def update_job_title(movie_id: int, req: TitleReq) -> dict[str, Any]:
    movie = db.get_movie(movie_id)
    if not movie:
        raise HTTPException(404, "Movie not found")
    title = req.title.strip()
    if not title:
        raise HTTPException(400, "title is required")
    fields: dict[str, Any] = {"collection_title": title}
    if req.update_original_name:
        fields["original_name"] = title
    db.update_movie(movie_id, **fields)
    return {"movie": db.get_movie(movie_id), "titles": db.list_collection_titles()}


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
def rerun_semantics_job(movie_id: int, req: SemanticsReq) -> dict[str, Any]:
    movie = db.get_movie(movie_id)
    if not movie:
        raise HTTPException(404, "Movie not found")
    if is_processing(movie_id):
        return {"started": False, "movie": movie, "message": "Movie is already running."}
    if db.count_clips({"movie_id": movie_id}) == 0:
        raise HTTPException(409, "No clips exist yet. Run the full job first.")
    try:
        profile = get_profile(req.profile_id)
        started = start_semantics_only(
            movie_id,
            profile.id,
            req.embeddings_per_clip,
            req.overwrite,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"started": started, "movie": db.get_movie(movie_id), "profile": profile.to_dict()}


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
    filter_query: Optional[str] = None,
    text: Optional[str] = None,
    semantic_text: Optional[str] = None,
    embedding_profile: str = DEFAULT_PROFILE_ID,
    embedding_mode: str = "mean_and_frames",
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
        "filter_query": filter_query,
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
    semantic_query = (semantic_text or "").strip()
    try:
        if semantic_query:
            rows = rank_semantic_clips(
                semantic_query,
                db.list_clips(filters),
                embedding_profile,
                embedding_mode,
            )
            total_count = len(rows)
            downloadable_count = sum(1 for row in rows if row.get("clip_path") and Path(row["clip_path"]).exists())
            if offset:
                rows = rows[max(0, int(offset)) :]
            if limit is not None:
                rows = rows[: max(0, int(limit))]
        else:
            rows = db.list_clips(filters, limit=limit, offset=offset)
            total_count = db.count_clips(filters)
            downloadable_count = db.count_clips({**filters, "has_file": True})
    except FilterQueryError as exc:
        raise HTTPException(400, str(exc)) from exc
    for row in rows:
        path = row.get("clip_path")
        row["size_mb"] = round(Path(path).stat().st_size / 1048576, 2) if path and Path(path).exists() else None
    return {
        "clips": rows,
        "count": total_count,
        "downloadable_count": downloadable_count,
        "semantic": bool(semantic_query),
        "embedding_profile": embedding_profile if semantic_query else None,
        "embedding_mode": embedding_mode if semantic_query else None,
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


@app.post("/semantic-match", dependencies=[Depends(auth)])
def semantic_match(req: SemanticMatchReq) -> dict[str, Any]:
    try:
        matches = _semantic_match(req)
    except FilterQueryError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "matches": matches,
        "count": len(matches),
        "profile_id": req.profile_id,
        "embedding_mode": req.embedding_mode,
    }


class BundleReq(BaseModel):
    movie_id: Optional[int] = None
    include_frames: bool = False


class ProfileBundleReq(BaseModel):
    movie_id: Optional[int] = None
    collection_title: Optional[str] = None
    include_clips: bool = True


@app.post("/embedding-profiles/{profile_id}/bundle", dependencies=[Depends(auth)])
def embedding_profile_bundle(profile_id: str, req: ProfileBundleReq) -> FileResponse:
    try:
        profile = get_profile(profile_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    filters: dict[str, Any] = {}
    if req.movie_id is not None:
        filters["movie_id"] = req.movie_id
    if req.collection_title:
        filters["collection_title"] = req.collection_title
    target_clips = [row for row in db.list_clips(filters) if row["status"] != "too_short"]
    records = {int(row["clip_id"]): row for row in db.list_clip_embeddings(profile_id, req.movie_id)}
    if req.collection_title:
        records = {
            clip_id: row
            for clip_id, row in records.items()
            if row.get("collection_title") == req.collection_title
        }
    missing = [
        int(clip["id"])
        for clip in target_clips
        if not records.get(int(clip["id"]))
        or records[int(clip["id"])].get("status") != "complete"
        or not Path(records[int(clip["id"])].get("artifact_path") or "").exists()
    ]
    if not target_clips:
        raise HTTPException(404, "No clips match this bundle request")
    if missing:
        raise HTTPException(
            409,
            detail={
                "code": "EMBEDDINGS_MISSING",
                "profile_id": profile_id,
                "complete": len(target_clips) - len(missing),
                "missing": len(missing),
                "message": "Generate this embedding profile in VastAI Program first.",
            },
        )

    export_dir = LIBRARY_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"movie-{req.movie_id}" if req.movie_id is not None else clean_filename(req.collection_title or "all", "all")
    out = export_dir / f"{profile_id}-{suffix}-{uuid.uuid4().hex[:8]}.zip"
    manifest: dict[str, Any] = {
        "format": "vastai-embedding-bundle-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile.to_dict(),
        "include_clips": req.include_clips,
        "clips": [],
    }
    # MP4 and NPZ are already compressed; storing avoids wasting CPU while TikTokGen waits.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for clip in target_clips:
            clip_id = int(clip["id"])
            record = records[clip_id]
            artifact = Path(record["artifact_path"])
            artifact_name = f"embeddings/clip_{clip_id:06d}.npz"
            archive.write(artifact, artifact_name)
            media_name = ""
            clip_path = Path(clip.get("clip_path") or "")
            if req.include_clips:
                if not clip_path.is_file():
                    raise HTTPException(409, f"Clip file is missing for clip {clip_id}")
                media_name = f"clips/clip_{clip_id:06d}{clip_path.suffix.lower() or '.mp4'}"
                archive.write(clip_path, media_name)
            manifest["clips"].append({
                "worker_clip_id": clip_id,
                "movie_id": int(clip["movie_id"]),
                "clip_index": int(clip["clip_index"]),
                "start_time": float(clip["start_time"]),
                "end_time": float(clip["end_time"]),
                "duration": float(clip["duration"]),
                "collection_title": record.get("collection_title") or "",
                "original_name": record.get("original_name") or "",
                "shot_size": clip.get("shot_size") or "unknown",
                "camera_motion_type": clip.get("camera_motion_type") or "unknown",
                "animation_motion_bucket": clip.get("animation_motion_bucket") or "unknown",
                "people_count": clip.get("people_count") or "unknown",
                "moods": clip.get("moods") or [],
                "tags": clip.get("tags") or [],
                "artifact": artifact_name,
                "media": media_name,
                "frame_count": int(record.get("frame_count") or 0),
                "dimension": int(record.get("dimension") or 0),
            })
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True, indent=2))
    return FileResponse(
        out,
        media_type="application/zip",
        filename=out.name,
        background=BackgroundTask(out.unlink, missing_ok=True),
    )


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


class CloudBackupReq(BaseModel):
    include_movies: bool = False
    include_frames: bool = False


class CloudRestoreReq(BaseModel):
    snapshot_id: str
    confirmation: str


@app.get("/backups", dependencies=[Depends(auth)])
def cloud_backups() -> dict[str, Any]:
    return {
        "rclone": rclone_ready(),
        "snapshots": list_snapshots(),
        "jobs": list(_CLOUD_JOBS.values()),
    }


@app.post("/backups", dependencies=[Depends(auth)])
def cloud_backup_start(req: CloudBackupReq) -> dict[str, Any]:
    job_id = start_cloud_job(
        "backup",
        create_snapshot,
        include_movies=req.include_movies,
        include_frames=req.include_frames,
    )
    return {"job_id": job_id, "laptop_may_disconnect": True}


@app.get("/backups/jobs/{job_id}", dependencies=[Depends(auth)])
def cloud_backup_job(job_id: str) -> dict[str, Any]:
    with _CLOUD_JOBS_LOCK:
        job = _CLOUD_JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Backup job not found")
        return dict(job)


@app.post("/backups/restore", dependencies=[Depends(auth)])
def cloud_restore_start(req: CloudRestoreReq) -> dict[str, Any]:
    expected = f"RESTORE {req.snapshot_id}"
    if req.confirmation != expected:
        raise HTTPException(400, f"confirmation must exactly equal {expected!r}")
    active = running_movie_ids()
    if active:
        raise HTTPException(409, f"Pause active movie jobs before restore: {active}")
    job_id = start_cloud_job("restore", restore_snapshot, req.snapshot_id)
    return {"job_id": job_id, "warning": "The library will switch atomically after integrity validation."}


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
