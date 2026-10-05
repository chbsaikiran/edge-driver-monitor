# Edge Driver Drowsiness & Distraction Alert System

In-cabin monitor that watches the driver through a single camera and raises a local alert
(speaker / buzzer + LED) when it sees fatigue or distraction. Everything runs on-device with
OpenCV and the MediaPipe Face Landmarker model.

| Alert        | Signal                                           | Default trigger                          |
|--------------|--------------------------------------------------|------------------------------------------|
| `DROWSY`     | Eye aspect ratio (EAR)                           | eyes closed for 1.5 s                    |
| `FATIGUE`    | PERCLOS (share of time eyes are closed)          | closed > 30% of the last 60 s            |
| `YAWN`       | Mouth aspect ratio (MAR)                         | mouth wide open for 1 s (LED only)       |
| `DISTRACTED` | Head pose (yaw / pitch) vs. calibrated neutral   | > 25° yaw or > 20° pitch for 2 s         |
| `NO_FACE`    | No face found                                    | 3 s                                      |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                      # install dependencies
uv run download_models.py    # fetch models/face_landmarker.task (~3.7 MB), once
```

## Run

```sh
uv run main.py                     # first camera showing a picture
uv run main.py --source 1          # a specific camera index
uv run main.py --source drive.mp4  # recorded video
uv run main.py --source csi --gpio --no-display   # Jetson CSI camera, buzzer + LED, headless
```

On start-up the system calibrates for 2 seconds: sit normally and look at the road. This
learns your neutral head pose (so a camera mounted low or off to the side works) and your
open-eye EAR. Press `c` in the preview window to recalibrate, `q` to quit.

On macOS, the first run asks for camera permission for your terminal / VS Code.

Run `uv run main.py --help` for all options (thresholds, pins, `--no-sound`, `--no-snapshots`).
Durations and other defaults live in [dms/config.py](dms/config.py).

## Output

- `logs/events.csv` — one row when each alert starts and one when it ends (with its duration),
  plus the EAR / MAR / head pose at that moment.
- `logs/snapshots/` — the camera frame that triggered each alert (disable with `--no-snapshots`).

## Moving to an edge board

- **USB camera**: `--source <index>`.
- **Jetson CSI camera**: `--source csi` (needs OpenCV built with GStreamer, as in JetPack).
- **Other CSI setups (e.g. Raspberry Pi libcamera)**: pass a GStreamer pipeline ending in
  `appsink` as `--source "libcamerasrc ! ... ! appsink"`.
- **Buzzer + LED**: `--gpio` with `--buzzer-pin` / `--led-pin` (BCM numbering, active-high;
  defaults 18 and 23). Install the board's GPIO library first: `uv add Jetson.GPIO` or
  `uv add RPi.GPIO` (`rpi-lgpio` on a Pi 5).

## Layout

```
main.py             camera loop, preview overlay, CLI
dms/face.py         MediaPipe landmarks -> EAR, MAR, head pose
dms/monitor.py      calibration + time-based alert logic (no camera/model dependency)
dms/alerts.py       speaker and GPIO outputs
dms/eventlog.py     CSV log + snapshots
dms/camera.py       USB / CSI / file sources
tests/              uv run pytest
```
