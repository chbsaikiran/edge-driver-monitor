"""Face landmarks (MediaPipe Face Landmarker, run locally) and the metrics derived from them."""

import math
import platform
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# MediaPipe face-mesh indices, ordered p1..p6 as in the EAR paper (Soukupova & Cech, 2016):
# p1/p4 are the eye corners, p2/p6 and p3/p5 the upper/lower lid pairs.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
# Inner lip contour: corners, then (upper, lower) pairs.
MOUTH_CORNERS = (78, 308)
MOUTH_VERTICALS = ((81, 178), (13, 14), (311, 402))


@dataclass
class FaceMetrics:
    ear: float  # mean of both eyes
    mar: float
    yaw: float  # degrees, + = nose towards image right
    pitch: float  # degrees, + = nose down
    roll: float  # degrees
    landmarks: np.ndarray  # (N, 2) pixel coordinates


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the face landmarker model on first use; afterwards everything runs offline."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face landmarker model to {path} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(path)
    return path


def eye_aspect_ratio(pts: np.ndarray, idx=RIGHT_EYE) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    horizontal = np.linalg.norm(pts[MOUTH_CORNERS[0]] - pts[MOUTH_CORNERS[1]])
    if horizontal < 1e-6:
        return 0.0
    vertical = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in MOUTH_VERTICALS])
    return float(vertical / horizontal)


def head_pose_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) from MediaPipe's 4x4 facial transformation matrix."""
    r = np.asarray(matrix, dtype=float)[:3, :3]
    r = r / np.linalg.norm(r, axis=0)  # drop any scale
    sy = math.hypot(r[0, 0], r[1, 0])
    yaw = math.atan2(-r[2, 0], sy)
    pitch = math.atan2(r[2, 1], r[2, 2])
    roll = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class FaceAnalyzer:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, delegate: str = "auto"):
        if delegate == "auto":
            # MediaPipe 1.x aborts on macOS with the CPU delegate (its graph still asks for
            # Metal), so use the GPU there. Elsewhere CPU (XNNPACK) works on any board.
            delegate = "gpu" if platform.system() == "Darwin" else "cpu"
        self._gpu = delegate == "gpu"
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(ensure_model(model_path)),
                delegate=BaseOptions.Delegate.GPU if self._gpu else BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray, t: float) -> FaceMetrics | None:
        """Analyse one frame taken at time t (seconds). Returns None when no face is found."""
        # VIDEO mode requires strictly increasing timestamps.
        ts_ms = max(int(t * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        # The GPU path only accepts 4-channel images.
        if self._gpu:
            data, fmt = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA), mp.ImageFormat.SRGBA
        else:
            data, fmt = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), mp.ImageFormat.SRGB
        image = mp.Image(image_format=fmt, data=data)
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return None

        h, w = frame_bgr.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]], dtype=np.float32)
        ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
        yaw, pitch, roll = head_pose_degrees(result.facial_transformation_matrixes[0])
        return FaceMetrics(ear, mouth_aspect_ratio(pts), yaw, pitch, roll, pts)

    def close(self) -> None:
        self._landmarker.close()
