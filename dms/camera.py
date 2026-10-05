"""Video sources: laptop/USB camera index, Jetson CSI camera, video file, or a GStreamer pipeline."""

import time

import cv2

WARMUP_SECONDS = 3.0
BLACK_FRAME_MEAN = 5.0


def jetson_csi_pipeline(width: int, height: int, fps: int = 30, sensor_id: int = 0) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


def _open(source: str, width: int, height: int) -> tuple[cv2.VideoCapture, bool]:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap, False
    if source == "csi":
        return cv2.VideoCapture(jetson_csi_pipeline(width, height), cv2.CAP_GSTREAMER), False
    if "!" in source:
        return cv2.VideoCapture(source, cv2.CAP_GSTREAMER), False
    return cv2.VideoCapture(source), True


def open_source(source: str, width: int, height: int, attempts: int = 4) -> tuple[cv2.VideoCapture, bool]:
    """Returns (capture, is_file).

    `source` is "auto", a camera index, "csi", a file path or a GStreamer pipeline.
    """
    if source == "auto":
        return _open_first_live_camera(width, height), False
    for _ in range(attempts):
        cap, is_file = _open(source, width, height)
        if cap.isOpened() and is_file:
            return cap, True
        # A camera can report "opened" yet deliver nothing while it powers up (or, on macOS,
        # while the permission prompt is showing), and may need reopening to recover.
        deadline = time.monotonic() + WARMUP_SECONDS
        while cap.isOpened() and time.monotonic() < deadline:
            if cap.read()[0]:
                return cap, False
            time.sleep(0.05)
        cap.release()
        if is_file:
            break
    raise RuntimeError(
        f"Could not get frames from video source {source!r}. For a camera, check it is connected, "
        "not in use by another app, and that this terminal has camera permission."
    )


def _open_first_live_camera(width: int, height: int, max_index: int = 4) -> cv2.VideoCapture:
    """Pick the first camera showing a real picture.

    Index 0 is not always the built-in camera: on macOS an iPhone (Continuity Camera) can take
    that slot and deliver nothing or black frames while the phone is asleep.
    """
    fallback = None
    for index in range(max_index):
        try:
            cap, _ = open_source(str(index), width, height, attempts=1)
        except RuntimeError:
            continue
        # Give auto-exposure a few frames before judging the picture as black.
        if any(ok and frame.mean() > BLACK_FRAME_MEAN for ok, frame in (cap.read() for _ in range(15))):
            if fallback is not None:
                fallback.release()
            print(f"Using camera {index}.")
            return cap
        if fallback is None:
            fallback = cap
        else:
            cap.release()
    if fallback is not None:
        print("Warning: every camera is delivering black frames; using the first one.")
        return fallback
    raise RuntimeError(
        "No working camera found. Check one is connected, not in use by another app, and that "
        "this terminal has camera permission."
    )
