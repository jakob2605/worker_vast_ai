from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionMetrics:
    camera_motion_type: str
    camera_motion_score: float
    translation_px_sec: float
    rotation_deg_sec: float
    zoom_delta: float
    confidence: float
    animation_motion_score: float
    animation_motion_bucket: str


def analyze_motion(video_path: Path, start_time: float, end_time: float, sample_count: int = 6, resize_width: int = 480) -> MotionMetrics:
    frames = sample_frames(video_path, start_time, end_time, sample_count, resize_width)
    if len(frames) < 2:
        return MotionMetrics("unknown", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "unknown")

    translations: list[float] = []
    rotations: list[float] = []
    zooms: list[float] = []
    residuals: list[float] = []
    confidences: list[float] = []

    seconds_between = max(0.001, (end_time - start_time) / max(1, len(frames) - 1))
    for prev, curr in zip(frames, frames[1:]):
        transform, confidence = _estimate_global_transform(prev, curr)
        translation, rotation, zoom = _decompose_transform(transform)
        residual = _residual_motion(prev, curr, transform)
        translations.append(translation / seconds_between)
        rotations.append(abs(rotation) / seconds_between)
        zooms.append(abs(zoom - 1.0))
        residuals.append(residual)
        confidences.append(confidence)

    translation_px_sec = float(np.median(translations)) if translations else 0.0
    rotation_deg_sec = float(np.median(rotations)) if rotations else 0.0
    zoom_delta = float(np.median(zooms)) if zooms else 0.0
    confidence = float(np.median(confidences)) if confidences else 0.0
    animation_score = float(np.median(residuals)) if residuals else 0.0
    camera_score = min(1.0, (translation_px_sec / 80.0) + (rotation_deg_sec / 20.0) + (zoom_delta * 8.0))

    motion_type = _classify_camera_motion(translation_px_sec, rotation_deg_sec, zoom_delta, confidence)
    return MotionMetrics(
        motion_type,
        round(camera_score, 4),
        round(translation_px_sec, 3),
        round(rotation_deg_sec, 3),
        round(zoom_delta, 5),
        round(confidence, 4),
        round(animation_score, 4),
        _bucket_animation_motion(animation_score),
    )


def sample_frames(video_path: Path, start_time: float, end_time: float, sample_count: int, resize_width: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = max(0.01, end_time - start_time)
    times = np.linspace(start_time + duration * 0.12, end_time - duration * 0.12, max(2, sample_count))
    frames: list[np.ndarray] = []
    for timestamp in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(timestamp * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        if resize_width and frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames


def _anime_preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    mixed = cv2.addWeighted(gray, 0.72, edges, 0.28, 0)
    return cv2.equalizeHist(mixed)


def _estimate_global_transform(prev: np.ndarray, curr: np.ndarray) -> tuple[np.ndarray | None, float]:
    prev_gray = _anime_preprocess(prev)
    curr_gray = _anime_preprocess(curr)

    detector = _create_feature_detector()
    kp1, des1 = detector.detectAndCompute(prev_gray, None)
    kp2, des2 = detector.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return _ecc_transform(prev_gray, curr_gray), 0.25

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return _ecc_transform(prev_gray, curr_gray), min(0.35, len(good) / 20)

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None or inliers is None:
        return _ecc_transform(prev_gray, curr_gray), 0.3
    confidence = float(np.count_nonzero(inliers)) / max(1, len(good))
    return matrix, confidence


def _create_feature_detector():
    akaze_factory = getattr(cv2, "AKAZE_create", None)
    if callable(akaze_factory):
        try:
            return akaze_factory()
        except (AttributeError, cv2.error):
            pass

    orb_factory = getattr(cv2, "ORB_create", None)
    if callable(orb_factory):
        return orb_factory(nfeatures=2500, fastThreshold=7)
    raise RuntimeError("OpenCV has neither AKAZE nor ORB feature detectors available.")


def _ecc_transform(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray | None:
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(
            prev_gray.astype(np.float32) / 255.0,
            curr_gray.astype(np.float32) / 255.0,
            warp,
            cv2.MOTION_AFFINE,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5),
        )
        return warp
    except cv2.error:
        return None


def _decompose_transform(matrix: np.ndarray | None) -> tuple[float, float, float]:
    if matrix is None:
        return 0.0, 0.0, 1.0
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    scale_x = (a * a + c * c) ** 0.5
    scale_y = (b * b + d * d) ** 0.5
    zoom = float((scale_x + scale_y) / 2.0)
    rotation = float(np.degrees(np.arctan2(c, a)))
    translation = float((tx * tx + ty * ty) ** 0.5)
    return translation, rotation, zoom


def _residual_motion(prev: np.ndarray, curr: np.ndarray, matrix: np.ndarray | None) -> float:
    prev_gray = _anime_preprocess(prev)
    curr_gray = _anime_preprocess(curr)
    if matrix is not None:
        curr_gray = cv2.warpAffine(curr_gray, matrix, (curr_gray.shape[1], curr_gray.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff) / 255.0)


def _classify_camera_motion(translation_px_sec: float, rotation_deg_sec: float, zoom_delta: float, confidence: float) -> str:
    if confidence < 0.18:
        return "unknown"
    if translation_px_sec < 6.0 and rotation_deg_sec < 0.4 and zoom_delta < 0.003:
        return "static"
    if zoom_delta >= 0.012:
        return "zoom"
    if rotation_deg_sec >= 3.0:
        return "handheld"
    if translation_px_sec >= 55.0:
        return "fast_pan"
    if translation_px_sec >= 10.0:
        return "slow_pan"
    if rotation_deg_sec >= 0.8:
        return "tilt"
    return "tracking"


def _bucket_animation_motion(score: float) -> str:
    if score < 0.025:
        return "none"
    if score < 0.07:
        return "low"
    if score < 0.14:
        return "medium"
    return "high"
