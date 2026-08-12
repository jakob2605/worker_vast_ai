# GPU worker

The clip pipeline from `Movie_To_Clips_Program`, ported to run on a rented Vast.ai GPU
instance and driven over HTTP from the local dashboard. **Code only — no library data was
copied.** Your existing local app is untouched and still works exactly as before.

## What changed from the original

| Stage | Original | Here |
|-------|----------|------|
| Import | `UploadFile` from the browser | `ingest_url()` — the box downloads the movie itself at datacenter bandwidth |
| Shot detection | `TransNetV2(device="cpu")` | `device=SETTINGS.device` → CUDA when present |
| Clip export | `libx264 -preset slow` | `h264_nvenc` on the GPU, automatic fallback to libx264 |
| Semantics | SigLIP2 on CPU, no `.to()` | `.to("cuda")`, optional fp16, batched |
| Motion | OpenCV, CPU | unchanged — it's cheap and CPU-bound anyway |
| Control | local FastAPI + browser | `worker.py` HTTP API behind a shared token |
| Storage | `library/` next to the app | `LIBRARY_DIR` env, default `/workspace/library` |

`db.py`, `motion.py` and the detection/labelling logic are byte-for-byte your originals
apart from the device lines. A migration adds the new `source_url`, `encoder`, `device`
columns to databases created by the old version.

## Files

```
worker.py            HTTP API that runs on the instance
bootstrap.sh         on-start script: installs ffmpeg + deps, launches the worker
requirements.txt     worker deps (torch comes from the vastai/pytorch image)
pipeline/            your modules, GPU-adapted
```

## How it gets onto the box

The dashboard's **Fill launch form for GPU worker** button sets the image, the
`-p 8100:8100` port mapping, a 200 GB disk and an on-start command that carries
`bootstrap.sh` gzipped+base64 (1.4 KB — under the API's ~4 KB onstart cap).

`bootstrap.sh` installs ffmpeg and the Python deps, then starts the worker. The
`pipeline/` code still has to reach the box — after the instance is running:

```powershell
vastai copy "C:\Users\jakob\Downloads\Vast_AI_Program\worker" <INSTANCE_ID>:/workspace/worker
```

Then re-run the bootstrap (or just restart the instance). The **Check worker** button
shows this exact command when it can't reach the worker.

## Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `WORKER_TOKEN` | *(unset)* | Shared secret. **Set it** — the port is public |
| `LIBRARY_DIR` | `/workspace/library` | Put this on a volume to survive destroy |
| `WORKER_PORT` | 8100 | Must match the `-p` mapping |
| `FORCE_DEVICE` | auto | `cpu` or `cuda` to override detection |
| `USE_NVENC` | on when CUDA | GPU encoding |
| `SIGLIP_BATCH` | 32 on GPU, 5 on CPU | Frames per forward pass |
| `SIGLIP_FP16` | on when CUDA | Half precision |

## API

```
GET  /health                 no auth — device, GPU, NVENC, disk
GET  /gpu                    model load state and device
POST /jobs {urls[]}          download + process
GET  /jobs                   progress per movie
POST /jobs/{id}/start|pause  resume control (your original pause flags)
GET  /clips?text=&shot_size= search the remote library
GET  /clips/{id}/file        pull ONE clip back
POST /bundle                 zip of sqlite + metadata + embeddings (no video)
GET  /storage                disk used by movies/clips/frames/embeddings
POST /purge                  delete source movies, keep clips
```

## Verified

Run end-to-end on CPU against a real 3-cut video: download by URL → 3 shots detected →
3 MP4s exported → motion classified → all 3 indexed → `complete`. Single-clip pull and
the metadata bundle both verified, and the bundle was confirmed to contain no video.
SigLIP fell back to the heuristic path in that test because torch wasn't installed —
that's your existing fallback behaving correctly, and the pytorch image supplies it.

## Cost note

Source movies are the bulk of the disk. Once clips exist, **Disk usage** then `/purge`
frees that space without touching your clips — worth doing between movies rather than
renting a bigger disk.
