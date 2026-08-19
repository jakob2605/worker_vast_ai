from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import traceback

import cv2
import numpy as np


@dataclass(frozen=True)
class Shot:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


def detect_shots(video_path: Path, fps: float, duration: float, threshold: float, merge_tiny_seconds: float) -> tuple[list[Shot], str]:
    for detector in (_transnet_detector, _pyscenedetect_detector, _opencv_histogram_detector):
        print(
            f"SHOT_DETECTOR_START detector={detector.__name__} "
            f"path={video_path} duration={duration:.3f}",
            flush=True,
        )
        try:
            shots, name = detector(video_path, fps, duration, threshold)
            if shots:
                merged = _merge_tiny_shots(shots, merge_tiny_seconds)
                print(
                    f"SHOT_DETECTOR_SUCCESS detector={detector.__name__} "
                    f"name={name} shots={len(merged)} path={video_path}",
                    flush=True,
                )
                return merged, name
            print(
                f"SHOT_DETECTOR_EMPTY detector={detector.__name__} "
                f"path={video_path} duration={duration:.3f}",
                flush=True,
            )
        except Exception:
            print(
                f"SHOT_DETECTOR_ERROR detector={detector.__name__} "
                f"path={video_path} duration={duration:.3f}",
                flush=True,
            )
            traceback.print_exc()
            continue
    fallback = [Shot(0, int(duration * fps), 0.0, duration)]
    print(
        f"SHOT_DETECTOR_FALLBACK path={video_path} duration={duration:.3f}",
        flush=True,
    )
    return fallback, "single-shot-fallback"


def _transnet_detector(video_path: Path, fps: float, duration: float, threshold: float) -> tuple[list[Shot], str]:
    from transnetv2_pytorch import TransNetV2

    from .config import SETTINGS

    # Was hardcoded to cpu. This is the first of the three GPU wins.
    model = TransNetV2(device=SETTINGS.device)
    model.eval()
    scenes = model.detect_scenes(str(video_path), threshold=threshold)
    shots: list[Shot] = []
    for scene in scenes:
        start_time = float(scene.get("start_time", 0.0))
        end_time = float(scene.get("end_time", start_time))
        start_frame = int(scene.get("start_frame", round(start_time * fps)))
        end_frame = int(scene.get("end_frame", round(end_time * fps)))
        if end_time > start_time:
            shots.append(Shot(start_frame, end_frame, start_time, end_time))
    return shots, f"transnetv2-pytorch-{SETTINGS.device}"


def _pyscenedetect_detector(video_path: Path, fps: float, duration: float, threshold: float) -> tuple[list[Shot], str]:
    from scenedetect import AdaptiveDetector, SceneManager, VideoManager

    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=max(2.0, threshold * 8.0)))
    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
    finally:
        video_manager.release()

    shots: list[Shot] = []
    for start, end in scene_list:
        start_time = start.get_seconds()
        end_time = end.get_seconds()
        shots.append(Shot(start.get_frames(), end.get_frames(), start_time, end_time))
    if not shots and duration:
        shots.append(Shot(0, int(duration * fps), 0.0, duration))
    return shots, "pyscenedetect-adaptive"


def _opencv_histogram_detector(video_path: Path, fps: float, duration: float, threshold: float) -> tuple[list[Shot], str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    cuts = [0]
    prev_hist = None
    frame_index = 0
    min_gap = max(1, int(fps * 0.5))
    sensitivity = max(0.35, min(0.85, 1.0 - threshold * 0.7))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < sensitivity and frame_index - cuts[-1] >= min_gap:
                cuts.append(frame_index)
        prev_hist = hist
        frame_index += 1
    cap.release()

    total_frames = max(frame_index, int(duration * fps))
    if cuts[-1] != total_frames:
        cuts.append(total_frames)

    shots = [
        Shot(cuts[i], cuts[i + 1], cuts[i] / fps if fps else 0.0, cuts[i + 1] / fps if fps else duration)
        for i in range(len(cuts) - 1)
        if cuts[i + 1] > cuts[i]
    ]
    return shots, "opencv-histogram-fallback"


def _merge_tiny_shots(shots: list[Shot], min_seconds: float) -> list[Shot]:
    if not shots:
        return []
    merged: list[Shot] = []
    current = shots[0]
    for shot in shots[1:]:
        if current.duration < min_seconds:
            current = Shot(current.start_frame, shot.end_frame, current.start_time, shot.end_time)
        else:
            merged.append(current)
            current = shot
    if current.duration < min_seconds and merged:
        prev = merged.pop()
        current = Shot(prev.start_frame, current.end_frame, prev.start_time, current.end_time)
    merged.append(current)
    return merged
