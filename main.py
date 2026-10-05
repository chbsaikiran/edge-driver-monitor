"""Edge driver drowsiness & distraction alert system.

    uv run main.py                 # first working laptop / USB camera
    uv run main.py --source 1      # another USB camera
    uv run main.py --source csi --gpio --no-display   # Jetson CSI camera, buzzer + LED, headless

Keys (when the preview window is focused): q = quit, c = recalibrate neutral pose.
"""

import argparse
import time
from pathlib import Path

import cv2

from dms.alerts import Alerter, GpioAlert, SoundAlert
from dms.camera import open_source
from dms.config import Config
from dms.eventlog import EventLogger
from dms.face import DEFAULT_MODEL_PATH, LEFT_EYE, MOUTH_CORNERS, MOUTH_VERTICALS, RIGHT_EYE, FaceAnalyzer
from dms.monitor import DriverMonitor

GREEN, RED, YELLOW, WHITE = (0, 220, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255)
MAX_DROPOUT_SECONDS = 2.0
MOUTH_POINTS = MOUTH_CORNERS + tuple(i for pair in MOUTH_VERTICALS for i in pair)


def parse_args() -> argparse.Namespace:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="auto", help='"auto" (first working camera), camera index, "csi" (Jetson), video file, or GStreamer pipeline')
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--delegate", choices=["auto", "cpu", "gpu"], default="auto", help="inference backend")
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument("--no-snapshots", action="store_true", help="don't save a camera frame with each alert")
    p.add_argument("--no-display", action="store_true", help="headless mode (no preview window)")
    p.add_argument("--no-sound", action="store_true", help="disable the speaker beep")
    p.add_argument("--gpio", action="store_true", help="drive a buzzer and LED over GPIO (Jetson / Raspberry Pi)")
    p.add_argument("--buzzer-pin", type=int, default=cfg.buzzer_pin, help="BCM pin number")
    p.add_argument("--led-pin", type=int, default=cfg.led_pin, help="BCM pin number")
    p.add_argument("--ear-ratio", type=float, default=cfg.ear_calib_ratio,
                   help="eyes count as closed below this fraction of the calibrated open-eye EAR")
    p.add_argument("--mar-threshold", type=float, default=cfg.mar_threshold)
    p.add_argument("--yaw-limit", type=float, default=cfg.yaw_limit_deg)
    p.add_argument("--pitch-limit", type=float, default=cfg.pitch_limit_deg)
    return p.parse_args()


def draw_overlay(frame, metrics, status, fps: float):
    """Mirrored preview (like a mirror) with landmarks, live metrics and the alert banner."""
    view = cv2.flip(frame, 1)
    h, w = view.shape[:2]

    if metrics is not None:
        for i in LEFT_EYE + RIGHT_EYE + MOUTH_POINTS:
            x, y = metrics.landmarks[i]
            cv2.circle(view, (int(w - 1 - x), int(y)), 1, GREEN, -1)
        lines = [
            f"EAR {metrics.ear:.2f} (thr {status.ear_threshold:.2f})",
            f"MAR {metrics.mar:.2f}",
            f"Yaw {status.yaw:+.0f}  Pitch {status.pitch:+.0f}",
            f"PERCLOS {status.perclos:.0%}",
        ]
    else:
        lines = ["No face"]
    lines.append(f"{fps:.0f} FPS")
    for n, text in enumerate(lines):
        cv2.putText(view, text, (10, 22 + 20 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    if status.calibrating:
        banner, color = "CALIBRATING - look at the road", YELLOW
    elif status.active:
        banner, color = " + ".join(sorted(status.active)), RED
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), RED, 8)  # on-screen stand-in for the LED
    else:
        banner, color = "OK", GREEN
    cv2.putText(view, banner, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return view


def main() -> None:
    args = parse_args()
    cfg = Config(
        ear_calib_ratio=args.ear_ratio,
        mar_threshold=args.mar_threshold,
        yaw_limit_deg=args.yaw_limit,
        pitch_limit_deg=args.pitch_limit,
    )

    analyzer = FaceAnalyzer(args.model, args.delegate)
    cap, is_file = open_source(args.source, args.width, args.height)
    monitor = DriverMonitor(cfg)
    logger = EventLogger(args.log_dir, snapshots=not args.no_snapshots)
    outputs = []
    if not args.no_sound:
        outputs.append(SoundAlert())
    if args.gpio:
        outputs.append(GpioAlert(args.buzzer_pin, args.led_pin))
    alerter = Alerter(outputs)

    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    last_frame_time = None
    fps = 0.0
    prev = time.monotonic()
    print(f"Running. Events are logged to {logger.path}. Press q in the preview (or Ctrl+C) to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    break
                # Ride out brief dropouts (USB hiccup, camera renegotiating) before giving up.
                if last_frame_time is None:
                    last_frame_time = time.monotonic()
                if time.monotonic() - last_frame_time > MAX_DROPOUT_SECONDS:
                    raise RuntimeError("Camera stopped delivering frames.")
                time.sleep(0.02)
                continue
            last_frame_time = time.monotonic()
            # Files use their own clock so timings stay right when processed faster than real time.
            t = frame_idx / file_fps if is_file else time.monotonic()
            frame_idx += 1

            metrics = analyzer.process(frame, t)
            status = monitor.update(metrics, t)
            alerter.update(status.active)
            for event in status.events:
                logger.log(event, metrics, status, frame)
                suffix = f" ({event.duration:.1f}s)" if event.phase == "end" else ""
                print(f"[{time.strftime('%H:%M:%S')}] {event.kind} {event.phase}{suffix}")

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1.0 / max(now - prev, 1e-6)
            prev = now

            if not args.no_display:
                cv2.imshow("Driver Monitor", draw_overlay(frame, metrics, status, fps))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    monitor.recalibrate()
    except KeyboardInterrupt:
        pass
    finally:
        alerter.close()
        logger.close()
        cap.release()
        analyzer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
