"""Append-only CSV event log, with an optional snapshot of the frame that triggered each alert."""

import csv
from datetime import datetime
from pathlib import Path

import cv2

FIELDS = ["timestamp", "event", "phase", "duration_s", "ear", "mar", "yaw_deg", "pitch_deg", "snapshot"]


class EventLogger:
    def __init__(self, log_dir: Path, snapshots: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._snap_dir = self.log_dir / "snapshots" if snapshots else None
        if self._snap_dir:
            self._snap_dir.mkdir(exist_ok=True)
        self.path = self.log_dir / "events.csv"
        new_file = not self.path.exists()
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        if new_file:
            self._writer.writeheader()

    def log(self, event, metrics, status, frame=None) -> None:
        now = datetime.now()
        snapshot = ""
        if self._snap_dir and frame is not None and event.phase == "start":
            snap_path = self._snap_dir / f"{now:%Y%m%d_%H%M%S_%f}_{event.kind}.jpg"
            if cv2.imwrite(str(snap_path), frame):
                snapshot = str(snap_path.relative_to(self.log_dir))
        self._writer.writerow(
            {
                "timestamp": now.isoformat(timespec="milliseconds"),
                "event": event.kind,
                "phase": event.phase,
                "duration_s": f"{event.duration:.2f}",
                "ear": f"{metrics.ear:.3f}" if metrics else "",
                "mar": f"{metrics.mar:.3f}" if metrics else "",
                "yaw_deg": f"{status.yaw:.1f}" if metrics else "",
                "pitch_deg": f"{status.pitch:.1f}" if metrics else "",
                "snapshot": snapshot,
            }
        )
        self._file.flush()  # survive a power cut on an edge device

    def close(self) -> None:
        self._file.close()
