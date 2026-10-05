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

All triggers are durations in seconds, so behaviour does not depend on the frame rate.
`FATIGUE` needs 30 s of history before it can fire. While no face is visible the other alerts
are ended, because the eyes and head can't be judged.

## Setup

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12+ and the dependencies.

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
open-eye EAR. Eyes then count as closed below 75% of that EAR (`--ear-ratio`), kept within
0.15–0.27. Press `c` in the preview window to recalibrate, `q` to quit.

On macOS, the first run asks for camera permission for your terminal / VS Code.

### Options

| Option | Default | Purpose |
|---|---|---|
| `--source` | `auto` | camera index, `csi`, video file or GStreamer pipeline |
| `--width`, `--height` | 640, 480 | requested capture size |
| `--model` | `models/face_landmarker.task` | face landmarker bundle, e.g. the int8 one (below) |
| `--delegate` | `auto` | inference backend: `cpu`, `gpu`, or `auto` (GPU on macOS, CPU elsewhere) |
| `--ear-ratio`, `--mar-threshold`, `--yaw-limit`, `--pitch-limit` | 0.75, 0.60, 25, 20 | alert thresholds |
| `--no-display` | off | headless, no preview window |
| `--no-sound` | off | no speaker beep |
| `--gpio`, `--buzzer-pin`, `--led-pin` | off, 18, 23 | buzzer and LED over GPIO |
| `--log-dir` | `logs` | where the event log and snapshots go |
| `--no-snapshots` | off | don't save a camera frame with each alert |

Durations (how long a condition must hold) are not on the command line; change them in
[dms/config.py](dms/config.py).

## Output

- `logs/events.csv` — one row when each alert starts and one when it ends (with its duration),
  plus the EAR / MAR / head pose at that moment.
- `logs/snapshots/` — the camera frame that triggered each alert (disable with `--no-snapshots`).

`logs/` holds pictures of the driver and is git-ignored.

## Moving to an edge board

- **USB camera**: `--source <index>`.
- **Jetson CSI camera**: `--source csi` (needs OpenCV built with GStreamer, as in JetPack).
- **Other CSI setups (e.g. Raspberry Pi libcamera)**: pass a GStreamer pipeline ending in
  `appsink` as `--source "libcamerasrc ! ... ! appsink"`.
- **Buzzer + LED**: `--gpio` with `--buzzer-pin` / `--led-pin` (BCM numbering, active-high;
  defaults 18 and 23). Install the board's GPIO library first: `uv add Jetson.GPIO` or
  `uv add RPi.GPIO` (`rpi-lgpio` on a Pi 5).

## Int8 model

[quantize/](quantize/) holds a three-step pipeline that turns the face detector and landmark
models inside `face_landmarker.task` into full-integer int8 models and repackages them as
`models/face_landmarker_int8.task`. It needs calibration images that you supply; see
[quantize/README.md](quantize/README.md).

```sh
uv run main.py --model models/face_landmarker_int8.task
```

## Tests

```sh
uv run pytest
```

The tests cover the alert logic with synthetic metrics, plus the EAR and head-pose maths. They
need no camera or model.

## Layout

```
main.py             camera loop, preview overlay, CLI
download_models.py  fetches the face landmarker model
dms/config.py       thresholds and durations
dms/face.py         MediaPipe landmarks -> EAR, MAR, head pose
dms/monitor.py      calibration + time-based alert logic (no camera/model dependency)
dms/alerts.py       speaker and GPIO outputs
dms/eventlog.py     CSV log + snapshots
dms/camera.py       USB / CSI / file sources
quantize/           int8 quantization of the models
tests/              uv run pytest
```
