# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "tensorflow==2.19.0",
#     "numpy<2.2",
# ]
# ///
"""Step 3: full-integer (int8) quantization with float32 inputs and outputs.

    uv run quantize/03_quantize_int8.py

Weights and all internal computation become int8; the models still take and return float32,
so MediaPipe's pre- and post-processing is unchanged. The result is packaged as
models/face_landmarker_int8.task, usable with `uv run main.py --model ...`.

Calibration data (see quantize/README.md) is read from:
    quantize/calib/face_detector/*.npy            float32, (128, 128, 3) or (N, 128, 128, 3), values in [-1, 1]
    quantize/calib/face_landmarks_detector/*.npy  float32, (256, 256, 3) or (N, 256, 256, 3), values in [0, 1]
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools import flatbuffer_utils

from common import CALIB, INT8, MODELS, ORIGINAL, SAVED_MODEL, TASK_FILE

MIN_SAMPLES = 100
METADATA_NAME = b"TFLITE_METADATA"


def load_calibration(folder: Path, size: int, value_range: tuple[float, float]) -> np.ndarray:
    files = sorted(folder.glob("*.npy"))
    if not files:
        sys.exit(f"No .npy calibration files in {folder}")
    batches = []
    for f in files:
        a = np.load(f)
        if a.ndim == 3:
            a = a[None]
        if a.shape[1:] != (size, size, 3):
            sys.exit(f"{f}: expected shape ({size}, {size}, 3) or (N, {size}, {size}, 3), got {a.shape}")
        batches.append(a.astype(np.float32))
    data = np.concatenate(batches)

    lo, hi = value_range
    if data.min() < lo - 0.01 or data.max() > hi + 0.01:
        sys.exit(
            f"{folder.name}: calibration values span [{data.min():.3g}, {data.max():.3g}] but the "
            f"model is fed [{lo:g}, {hi:g}] by MediaPipe. Fix the normalisation, otherwise the "
            "quantization ranges will be wrong."
        )
    if data.max() - data.min() < 0.5 * (hi - lo):
        print(f"  WARNING: values only span [{data.min():.3g}, {data.max():.3g}] of [{lo:g}, {hi:g}]; "
              "check the normalisation.")
    if len(data) < MIN_SAMPLES:
        print(f"  WARNING: only {len(data)} calibration samples; a few hundred varied ones are recommended.")
    return data


def quantize(saved_model: Path, calibration: np.ndarray) -> bytes:
    def representative_dataset():
        for sample in calibration:
            yield [sample[None]]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Restricting the op set makes conversion fail instead of silently leaving an op in float.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    return converter.convert()


def match_original(model_bytes: bytes, original: Path) -> bytes:
    """Make the quantized model a drop-in replacement for `original` inside the .task bundle."""
    model = flatbuffer_utils.convert_bytearray_to_object(bytearray(model_bytes))
    source = flatbuffer_utils.read_model(str(original))

    # MediaPipe reads outputs by position, so restore the original order.
    want = [tuple(d["shape"]) for d in tf.lite.Interpreter(str(original)).get_output_details()]
    have = [tuple(d["shape"]) for d in tf.lite.Interpreter(model_content=model_bytes).get_output_details()]
    if have != want:
        if sorted(have) != sorted(want) or len(set(want)) != len(want):
            sys.exit(f"Output shapes {have} cannot be matched to the original's {want}.")
        subgraph = model.subgraphs[0]
        by_shape = dict(zip(have, list(subgraph.outputs)))
        subgraph.outputs = [by_shape[shape] for shape in want]
    model.signatureDefs = None  # they name outputs in the converter's order; MediaPipe doesn't use them

    # MediaPipe refuses a float-input model without the TFLite metadata (input normalisation
    # etc.) that the original carries, and conversion dropped it. Copy it across.
    model.metadata = [m for m in model.metadata or [] if m.name != METADATA_NAME]
    for entry in source.metadata or []:
        if entry.name == METADATA_NAME:
            buffer = schema_fb.BufferT()
            buffer.data = source.buffers[entry.buffer].data
            model.buffers.append(buffer)
            copied = schema_fb.MetadataT()
            copied.name = METADATA_NAME
            copied.buffer = len(model.buffers) - 1
            model.metadata.append(copied)
    return bytes(flatbuffer_utils.convert_object_to_bytearray(model))


def run(interp_kwargs: dict, x: np.ndarray) -> list[np.ndarray]:
    interp = tf.lite.Interpreter(**interp_kwargs)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]


def report(name: str, model_bytes: bytes, original: Path, calibration: np.ndarray) -> None:
    interp = tf.lite.Interpreter(model_content=model_bytes)
    io = interp.get_input_details() + interp.get_output_details()
    if any(d["dtype"] != np.float32 for d in io):
        sys.exit(f"{name}: inputs/outputs are not all float32.")
    dtypes = {}
    for t in interp.get_tensor_details():
        dtypes[t["dtype"].__name__] = dtypes.get(t["dtype"].__name__, 0) + 1
    print(f"  tensors by type: {dtypes}")
    print(f"  size: {original.stat().st_size / 1e6:.2f} MB -> {len(model_bytes) / 1e6:.2f} MB")

    # Accuracy on up to 50 calibration samples, against the original float model.
    samples = calibration[:: max(1, len(calibration) // 50)][:50]
    errors = []
    for sample in samples:
        ref = run({"model_path": str(original)}, sample[None])
        out = run({"model_content": model_bytes}, sample[None])
        errors.append([float(np.abs(r - o).mean()) for r, o in zip(ref, out)])
        scale = [float(np.abs(r).max()) for r in ref]
    print(f"  mean |error| vs original, per output: {[f'{e:.3g}' for e in np.mean(errors, axis=0)]}"
          f"  (output magnitudes up to {[f'{s:.3g}' for s in scale]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib-dir", type=Path, default=CALIB)
    parser.add_argument("--output", type=Path, default=TASK_FILE.with_name("face_landmarker_int8.task"))
    args = parser.parse_args()

    INT8.mkdir(parents=True, exist_ok=True)
    for name, (size, value_range) in MODELS.items():
        print(f"\n{name}")
        calibration = load_calibration(args.calib_dir / name, size, value_range)
        print(f"  calibrating on {len(calibration)} samples ...")
        original = ORIGINAL / f"{name}.tflite"
        model_bytes = match_original(quantize(SAVED_MODEL / name, calibration), original)
        (INT8 / f"{name}.tflite").write_bytes(model_bytes)
        report(name, model_bytes, original, calibration)

    # Same bundle layout as the original: uncompressed entries, quantized models swapped in,
    # the unused blendshapes model and the geometry metadata copied as they are.
    with zipfile.ZipFile(TASK_FILE) as src, zipfile.ZipFile(args.output, "w", zipfile.ZIP_STORED) as dst:
        for entry in src.namelist():
            quantized = INT8 / entry
            dst.writestr(entry, quantized.read_bytes() if quantized.exists() else src.read(entry))
    print(f"\nWrote {args.output}\nTry it:  uv run main.py --model {args.output}")


if __name__ == "__main__":
    main()
