# Int8 quantization of the face landmarker models

Turns the two models inside `models/face_landmarker.task` that this project uses
(`face_detector.tflite`, `face_landmarks_detector.tflite`) into full-integer int8 models with
float32 inputs and outputs, and repackages them as `models/face_landmarker_int8.task`.

Each script declares its own dependencies and Python version, so `uv run` builds an isolated
environment for it; nothing is added to the main project. Run from the repository root.

```sh
uv run quantize/01_tflite_to_onnx.py   # unpack the bundle, TFLite -> ONNX
uv run quantize/02_onnx_to_tf.py       # ONNX -> TensorFlow SavedModel, then equivalence check
# ... put calibration data in quantize/calib/ (below) ...
uv run quantize/03_quantize_int8.py    # int8 quantization + new .task bundle
uv run main.py --model models/face_landmarker_int8.task
```

Intermediate files go to `quantize/work/`.

## Step 2's check

The same random inputs go through the original TFLite model, the ONNX model and the
SavedModel. A stage passes when every output is within `--tol` (default 1e-4) of the original,
measured relative to the output's scale. The "noise floor" row is the original model compared
with itself under a different kernel implementation: it shows how much difference is plain
float32 rounding.

## Calibration data

`.npy` files, float32, any file names, one or many samples per file:

| Folder | Shape | Values | What the model sees in MediaPipe |
|---|---|---|---|
| `quantize/calib/face_detector/` | `(128, 128, 3)` or `(N, 128, 128, 3)` | RGB in `[-1, 1]` (`pixel / 127.5 - 1`) | the whole camera frame, letterboxed (aspect ratio kept, padded with black) to 128x128 |
| `quantize/calib/face_landmarks_detector/` | `(256, 256, 3)` or `(N, 256, 256, 3)` | RGB in `[0, 1]` (`pixel / 255`) | a square crop around the face, about 1.5x the detected face box, rotated so the eyes are level, resized to 256x256 |

Notes:

- Channel order is RGB. OpenCV reads BGR, so convert.
- Use a few hundred samples per model that look like real use: eyes open and closed, yawns,
  head turned and tilted, glasses, day and night lighting, the actual camera and mounting
  position if possible.
- Crops for the landmark model should be framed like MediaPipe's, otherwise the calibrated
  ranges won't match what the model sees at run time.
- Step 3 refuses data outside the expected value range and warns about small sample counts.

## What step 3 does

1. Quantizes each SavedModel with `TFLiteConverter` (int8 weights and activations, float32
   input/output, conversion fails if any op would stay float).
2. Restores the original output order and copies the original's TFLite metadata into the new
   file; MediaPipe needs both.
3. Prints tensor types, size, and the mean output error against the original model on the
   calibration samples.
4. Writes the new `.task` bundle; the blendshapes model and geometry metadata are copied as is.
