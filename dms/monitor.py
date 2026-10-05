"""Turns per-frame face metrics into drowsiness / distraction alerts.

Pure logic with no camera or model dependency, so it can be unit tested with synthetic input.
"""

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from .config import Config

DROWSY = "DROWSY"  # eyes closed continuously (microsleep)
FATIGUE = "FATIGUE"  # eyes closed for a large share of the last minute (PERCLOS)
YAWN = "YAWN"
DISTRACTED = "DISTRACTED"  # head turned away from the road
NO_FACE = "NO_FACE"  # driver not visible


@dataclass
class Event:
    kind: str
    phase: str  # "start" or "end"
    duration: float  # seconds the alert lasted; 0 for "start"


@dataclass
class Status:
    calibrating: bool = False
    active: set[str] = field(default_factory=set)
    events: list[Event] = field(default_factory=list)
    ear_threshold: float = 0.0
    perclos: float = 0.0
    yaw: float = 0.0  # relative to the calibrated neutral pose
    pitch: float = 0.0


class _Sustained:
    """Reports True once a condition has held continuously for `hold` seconds."""

    def __init__(self, hold: float):
        self.hold = hold
        self._since: float | None = None

    def update(self, condition: bool, t: float) -> bool:
        if not condition:
            self._since = None
            return False
        if self._since is None:
            self._since = t
        return t - self._since >= self.hold

    def reset(self) -> None:
        self._since = None


class DriverMonitor:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self._eyes_closed = _Sustained(self.cfg.eye_closed_seconds)
        self._yawning = _Sustained(self.cfg.yawn_seconds)
        self._head_away = _Sustained(self.cfg.distraction_seconds)
        self._no_face = _Sustained(self.cfg.no_face_seconds)
        self._started: dict[str, float] = {}  # active alert -> start time
        self.recalibrate()

    def recalibrate(self) -> None:
        """Re-learn the driver's neutral head pose and open-eye EAR from the next few seconds."""
        self._calib_start: float | None = None
        self._calib_samples: list[tuple[float, float, float]] = []
        self._calibrated = False
        self._ear_threshold = self.cfg.ear_threshold
        self._yaw0 = 0.0
        self._pitch0 = 0.0
        self._closed_history: deque[tuple[float, bool]] = deque()
        for timer in (self._eyes_closed, self._yawning, self._head_away):
            timer.reset()

    def update(self, metrics, t: float) -> Status:
        """Feed one frame's metrics (or None when no face was found) taken at time t seconds."""
        cfg = self.cfg
        status = Status(ear_threshold=self._ear_threshold)
        conditions = {NO_FACE: self._no_face.update(metrics is None, t)}

        if metrics is None:
            # Can't tell what the eyes/head are doing; don't let stale timers fire later.
            for timer in (self._eyes_closed, self._yawning, self._head_away):
                timer.reset()
        elif not self._calibrated:
            self._calibrate(metrics, t)
            status.calibrating = not self._calibrated
        else:
            status.yaw = metrics.yaw - self._yaw0
            status.pitch = metrics.pitch - self._pitch0
            closed = metrics.ear < self._ear_threshold
            away = abs(status.yaw) > cfg.yaw_limit_deg or abs(status.pitch) > cfg.pitch_limit_deg

            conditions[DROWSY] = self._eyes_closed.update(closed, t)
            conditions[YAWN] = self._yawning.update(metrics.mar > cfg.mar_threshold, t)
            conditions[DISTRACTED] = self._head_away.update(away, t)
            status.perclos = self._perclos(closed, t)
            conditions[FATIGUE] = status.perclos > cfg.perclos_threshold

        status.ear_threshold = self._ear_threshold
        if metrics is None and not self._calibrated:
            status.calibrating = True

        for kind, on in conditions.items():
            if on and kind not in self._started:
                self._started[kind] = t
                status.events.append(Event(kind, "start", 0.0))
            elif not on and kind in self._started:
                status.events.append(Event(kind, "end", t - self._started.pop(kind)))
        # Alerts whose condition wasn't evaluated this frame (e.g. face lost) are ended too.
        for kind in [k for k in self._started if k not in conditions]:
            status.events.append(Event(kind, "end", t - self._started.pop(kind)))

        status.active = set(self._started)
        return status

    def _calibrate(self, metrics, t: float) -> None:
        if self._calib_start is None:
            self._calib_start = t
        self._calib_samples.append((metrics.ear, metrics.yaw, metrics.pitch))
        if t - self._calib_start < self.cfg.calibration_seconds:
            return
        ears, yaws, pitches = zip(*self._calib_samples)
        cfg = self.cfg
        self._ear_threshold = min(
            cfg.ear_threshold_max, max(cfg.ear_threshold_min, median(ears) * cfg.ear_calib_ratio)
        )
        self._yaw0 = median(yaws)
        self._pitch0 = median(pitches)
        self._calibrated = True

    def _perclos(self, closed: bool, t: float) -> float:
        window = self.cfg.perclos_window_seconds
        history = self._closed_history
        history.append((t, closed))
        while history and t - history[0][0] > window:
            history.popleft()
        # Wait until the window is at least half full so a single blink at start-up isn't 100%.
        if t - history[0][0] < window / 2:
            return 0.0
        return sum(c for _, c in history) / len(history)
