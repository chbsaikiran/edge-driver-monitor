# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "onnx2tf==1.28.8",
#     "tensorflow==2.19.0",
#     "tf-keras==2.19.0",
#     "onnx==1.17.0",
#     "onnxruntime",
#     "onnx-graphsurgeon",
#     "sng4onnx",
#     "ai-edge-litert",
#     "psutil",
#     "numpy<2.2",
# ]
# ///
"""Step 2: convert the ONNX models to TensorFlow SavedModels and verify nothing changed.

    uv run quantize/02_onnx_to_tf.py

The same inputs are run through the original TFLite model, the ONNX model and the SavedModel.
The script exits with an error if any output differs from the original by more than --tol
(measured relative to each output's scale; see worst_errors).
"""

import argparse
import os
import sys

import numpy as np
import onnx2tf
import onnxruntime as ort
import tensorflow as tf

from common import MODELS, ONNX, ORIGINAL, SAVED_MODEL, WORK

PROBE_FILE = "calibration_image_sample_data_20x128x128x3_float32.npy"


def run_tflite(path, x, reference_kernels=False):
    kwargs = {}
    if reference_kernels:
        kwargs["experimental_op_resolver_type"] = (
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    interp = tf.lite.Interpreter(str(path), **kwargs)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]


def run_onnx(path, x):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    nchw = np.transpose(x, (0, 3, 1, 2))
    return session.run(None, {session.get_inputs()[0].name: nchw})


def run_saved_model(path, x):
    fn = tf.saved_model.load(str(path)).signatures["serving_default"]
    return [v.numpy() for v in fn(tf.constant(x)).values()]


def worst_errors(reference, candidate):
    """For each reference output: (max |difference|, the same divided by the output's scale).

    The scale is max(1, largest |value| in that output). Landmark coordinates are in pixels
    (up to ~256), where float32 resolves about 3e-5 per operation, so after ~70 layers an
    absolute 1e-4 is below what the original model reproduces against itself; the scaled
    error is the one compared with --tol.

    Converters rename and reorder outputs, so counterparts are paired by element count and,
    where several have the same size, by closest values.
    """
    remaining = list(candidate)
    errors = []
    for ref in reference:
        scored = [
            (float(np.max(np.abs(ref.ravel() - c.ravel()))), i)
            for i, c in enumerate(remaining)
            if c.size == ref.size
        ]
        if not scored:
            errors.append((float("inf"), float("inf")))
            continue
        diff, i = min(scored)
        remaining.pop(i)
        errors.append((diff, diff / max(1.0, float(np.max(np.abs(ref))))))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tol", type=float, default=1e-4, help="max allowed output error, relative to the output's scale")
    parser.add_argument("--trials", type=int, default=5, help="random inputs to test per model")
    args = parser.parse_args()

    # onnx2tf probes each graph with a sample image batch that it otherwise downloads from
    # GitHub at conversion time. Supplying one locally keeps this step offline.
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    np.save(PROBE_FILE, np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32))

    failed = False
    for name, (size, (lo, hi)) in MODELS.items():
        onnx_path = ONNX / f"{name}.onnx"
        saved_model_path = SAVED_MODEL / name
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_model_path),
            output_signaturedefs=True,
            non_verbose=True,
        )

        original = ORIGINAL / f"{name}.tflite"
        rng = np.random.default_rng(0)
        # "noise floor": the original model against itself with a different kernel
        # implementation. Shown for context; it is not a pass/fail row.
        worst = {"noise floor": None, "onnx": None, "saved_model": None}
        for _ in range(args.trials):
            x = rng.uniform(lo, hi, size=(1, size, size, 3)).astype(np.float32)
            reference = run_tflite(original, x)
            runs = {
                "noise floor": run_tflite(original, x, reference_kernels=True),
                "onnx": run_onnx(onnx_path, x),
                "saved_model": run_saved_model(saved_model_path, x),
            }
            for stage, outputs in runs.items():
                errors = np.array(worst_errors(reference, outputs))
                worst[stage] = errors if worst[stage] is None else np.maximum(worst[stage], errors)

        print(f"\n{name}  output shapes: {[r.shape for r in reference]}")
        print(f"  {'':<4}  {'original tflite vs':<19}  max |diff| per output          scaled error per output")
        for stage, errors in worst.items():
            ok = errors[:, 1].max() <= args.tol
            verdict = "" if stage == "noise floor" else ("PASS" if ok else "FAIL")
            failed |= verdict == "FAIL"
            absolute = ", ".join(f"{e:.1e}" for e in errors[:, 0])
            scaled = ", ".join(f"{e:.1e}" for e in errors[:, 1])
            print(f"  {verdict:<4}  {stage:<19}  {absolute:<29}  {scaled}")

    if failed:
        sys.exit(f"\nFAILED: at least one output has a scaled error above {args.tol:g}.")
    print(f"\nAll outputs match the original models within {args.tol:g}. SavedModels are in {SAVED_MODEL}")


if __name__ == "__main__":
    main()
