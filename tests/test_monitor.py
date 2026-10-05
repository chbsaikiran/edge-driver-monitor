from types import SimpleNamespace

import numpy as np

from dms.config import Config
from dms.face import RIGHT_EYE, eye_aspect_ratio, head_pose_degrees
from dms.monitor import DISTRACTED, DROWSY, FATIGUE, NO_FACE, YAWN, DriverMonitor

FPS = 20


def face(ear=0.30, mar=0.05, yaw=0.0, pitch=0.0):
    return SimpleNamespace(ear=ear, mar=mar, yaw=yaw, pitch=pitch)


def run(monitor, t0, seconds, metrics):
    """Feed constant metrics for `seconds`; returns (end time, last status, all events)."""
    events, status = [], None
    for i in range(int(seconds * FPS)):
        status = monitor.update(metrics, t0 + i / FPS)
        events += status.events
    return t0 + seconds, status, events


def calibrated(**neutral):
    monitor = DriverMonitor(Config())
    t, status, _ = run(monitor, 0.0, 2.5, face(**neutral))
    assert not status.calibrating
    return monitor, t


def test_calibration_sets_personal_ear_threshold():
    monitor, t = calibrated(ear=0.32)
    status = monitor.update(face(ear=0.32), t)
    assert abs(status.ear_threshold - 0.24) < 1e-6
    assert not status.active


def test_blink_does_not_alert_but_long_closure_does():
    monitor, t = calibrated()
    t, status, _ = run(monitor, t, 0.3, face(ear=0.10))
    assert DROWSY not in status.active
    t, _, _ = run(monitor, t, 1.0, face())
    t, status, events = run(monitor, t, 2.0, face(ear=0.10))
    assert DROWSY in status.active
    assert [(e.kind, e.phase) for e in events] == [(DROWSY, "start")]
    _, status, events = run(monitor, t, 0.5, face())
    assert DROWSY not in status.active
    assert events[0].phase == "end" and 0.4 < events[0].duration < 0.7


def test_yawn():
    monitor, t = calibrated()
    _, status, _ = run(monitor, t, 1.5, face(mar=0.8))
    assert status.active == {YAWN}


def test_head_turn_is_relative_to_neutral_pose():
    # Laptop camera below the face: neutral pitch is far from zero and must not alert.
    monitor, t = calibrated(pitch=-18.0)
    t, status, _ = run(monitor, t, 3.0, face(pitch=-18.0))
    assert not status.active
    t, status, _ = run(monitor, t, 1.0, face(pitch=-18.0, yaw=40.0))
    assert not status.active  # a mirror check is shorter than the hold time
    _, status, _ = run(monitor, t, 1.5, face(pitch=-18.0, yaw=40.0))
    assert status.active == {DISTRACTED}


def test_no_face_alerts_and_ends_other_alerts():
    monitor, t = calibrated()
    t, status, _ = run(monitor, t, 2.0, face(ear=0.10))
    assert DROWSY in status.active
    _, status, events = run(monitor, t, 3.5, None)
    assert status.active == {NO_FACE}
    assert (DROWSY, "end") in [(e.kind, e.phase) for e in events]


def test_perclos_fatigue_from_frequent_long_blinks():
    monitor, t = calibrated()
    status = None
    for _ in range(40):  # eyes shut 40% of the time, never long enough for DROWSY
        t, _, _ = run(monitor, t, 0.6, face(ear=0.10))
        t, status, _ = run(monitor, t, 0.9, face())
    assert status.active == {FATIGUE}


def test_eye_aspect_ratio_geometry():
    pts = np.zeros((478, 2), dtype=np.float32)
    coords = [(0, 0), (1, 1), (2, 1), (3, 0), (2, -1), (1, -1)]
    for i, xy in zip(RIGHT_EYE, coords):
        pts[i] = xy
    assert abs(eye_aspect_ratio(pts, RIGHT_EYE) - 2 / 3) < 1e-6


def test_head_pose_from_rotation_matrix():
    a = np.radians(30)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    yaw, pitch, roll = head_pose_degrees(m * 2.0)  # scale must not matter
    assert abs(yaw - 30) < 1e-6 and abs(pitch) < 1e-6 and abs(roll) < 1e-6
