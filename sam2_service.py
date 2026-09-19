"""Local-only SAM2 GPU service in an isolated Python environment.

It deliberately shares no Python packages with worker.py.  The media worker
can keep its proven Torch 2.4 stack for SigLIP, Aesthetic and LanguageBind,
while this process owns SAM2's newer Torch runtime.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Local SAM2 service", docs_url=None, redoc_url=None)
_LOCK = threading.Lock()
_GENERATOR: Any | None = None
_MODEL: Any | None = None


class SegmentRequest(BaseModel):
    source: str
    frame_time: float = 0.0


class PromptRequest(SegmentRequest):
    point: list[float]


def _generator() -> Any:
    global _GENERATOR, _MODEL
    with _LOCK:
        if _GENERATOR is not None:
            return _GENERATOR
        checkpoint = Path(os.getenv("SAM2_CHECKPOINT", "/workspace/checkpoints/sam2.1_hiera_large.pt"))
        if not checkpoint.is_file():
            raise HTTPException(503, "SAM2 checkpoint is missing")
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            _MODEL = build_sam2(os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml"), str(checkpoint), device="cuda", apply_postprocessing=False)
            _GENERATOR = SAM2AutomaticMaskGenerator(_MODEL, points_per_side=20, points_per_batch=32, pred_iou_thresh=.84, stability_score_thresh=.91, min_mask_region_area=900)
        except Exception as exc:
            raise HTTPException(503, f"SAM2 could not load: {exc}") from exc
        return _GENERATOR


def _frame(source_text: str, frame_time: float) -> Any:
    import cv2
    from PIL import Image, ImageOps
    source = Path(source_text).resolve()
    library = Path(os.getenv("LIBRARY_DIR", "/workspace/library")).resolve()
    if not source.is_file() or library not in source.parents:
        raise HTTPException(400, "Invalid clip source")
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, frame_time) * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise HTTPException(422, "Could not extract requested frame")
    return ImageOps.fit(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (1080, 1920), method=Image.Resampling.LANCZOS)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/segment")
def segment(req: SegmentRequest) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    image = _frame(req.source, req.frame_time)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = _generator().generate(np.asarray(image))
    except Exception as exc:
        raise HTTPException(503, f"SAM2 inference failed: {exc}") from exc
    editable: list[dict[str, Any]] = []
    for index, item in enumerate(sorted(masks, key=lambda value: float(value.get("area", 0)), reverse=True)):
        if len(editable) >= 45: break
        contours, _ = cv2.findContours(item["segmentation"].astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 900: continue
            polygon = cv2.approxPolyDP(contour, max(2.0, .004 * cv2.arcLength(contour, True)), True).reshape(-1, 2).tolist()
            if len(polygon) >= 3: editable.append({"id": f"{index}-{len(editable)}", "polygon": polygon})
            if len(editable) >= 45: break
    return {"width": 1080, "height": 1920, "masks": editable}


@app.post("/prompt")
def prompt(req: PromptRequest) -> dict[str, Any]:
    """Return one editable contour for a positive SAM2 point prompt."""
    import cv2
    import numpy as np
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if len(req.point) != 2:
        raise HTTPException(422, "point must contain exactly x and y")
    image = _frame(req.source, req.frame_time)
    _generator()  # loads and retains the shared SAM2 model
    if _MODEL is None:
        raise HTTPException(503, "SAM2 model is unavailable")
    x, y = max(0.0, min(1079.0, float(req.point[0]))), max(0.0, min(1919.0, float(req.point[1])))
    try:
        with _LOCK, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor = SAM2ImagePredictor(_MODEL)
            predictor.set_image(np.asarray(image))
            masks, _scores, _logits = predictor.predict(
                point_coords=np.asarray([[x, y]], dtype=np.float32),
                point_labels=np.asarray([1], dtype=np.int32), multimask_output=False,
            )
    except Exception as exc:
        raise HTTPException(503, f"SAM2 point inference failed: {exc}") from exc
    mask = masks[0].astype("uint8") if len(masks) else None
    if mask is None or not mask.any():
        raise HTTPException(422, "SAM2 found no mask at that point")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise HTTPException(422, "SAM2 returned no contour")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 25:
        raise HTTPException(422, "SAM2 mask is too small")
    polygon = cv2.approxPolyDP(contour, max(2.0, .002 * cv2.arcLength(contour, True)), True).reshape(-1, 2).tolist()
    if len(polygon) < 3:
        raise HTTPException(422, "SAM2 contour is invalid")
    return {"id": f"point-{int(x)}-{int(y)}", "polygon": polygon, "width": 1080, "height": 1920}
