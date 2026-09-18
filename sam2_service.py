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


class SegmentRequest(BaseModel):
    source: str
    frame_time: float = 0.0


def _generator() -> Any:
    global _GENERATOR
    with _LOCK:
        if _GENERATOR is not None:
            return _GENERATOR
        checkpoint = Path(os.getenv("SAM2_CHECKPOINT", "/workspace/checkpoints/sam2.1_hiera_large.pt"))
        if not checkpoint.is_file():
            raise HTTPException(503, "SAM2 checkpoint is missing")
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            model = build_sam2(os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml"), str(checkpoint), device="cuda", apply_postprocessing=False)
            _GENERATOR = SAM2AutomaticMaskGenerator(model, points_per_side=20, points_per_batch=32, pred_iou_thresh=.84, stability_score_thresh=.91, min_mask_region_area=900)
        except Exception as exc:
            raise HTTPException(503, f"SAM2 could not load: {exc}") from exc
        return _GENERATOR


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/segment")
def segment(req: SegmentRequest) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    source = Path(req.source).resolve()
    library = Path(os.getenv("LIBRARY_DIR", "/workspace/library")).resolve()
    if not source.is_file() or library not in source.parents:
        raise HTTPException(400, "Invalid clip source")
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, req.frame_time) * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise HTTPException(422, "Could not extract requested frame")
    image = ImageOps.fit(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (1080, 1920), method=Image.Resampling.LANCZOS)
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
