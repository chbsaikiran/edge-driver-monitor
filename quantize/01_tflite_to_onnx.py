# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "tf2onnx==1.16.1",
#     "tensorflow==2.15.1",
#     "numpy<2",
#     "onnx<1.17",
#     "protobuf<4",
# ]
# ///
"""Step 1: unpack face_landmarker.task and convert the two models to ONNX.

    uv run quantize/01_tflite_to_onnx.py
"""

import subprocess
import sys
import zipfile

import tensorflow as tf

from common import MODELS, ONNX, ORIGINAL, TASK_FILE


def main() -> None:
    ORIGINAL.mkdir(parents=True, exist_ok=True)
    ONNX.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TASK_FILE) as bundle:
        bundle.extractall(ORIGINAL)

    for name in MODELS:
        src = ORIGINAL / f"{name}.tflite"
        dst = ONNX / f"{name}.onnx"
        input_name = tf.lite.Interpreter(str(src)).get_input_details()[0]["name"]
        # Channels-first input is what ONNX tools expect; step 2 turns it back into NHWC
        # without leaving stray transposes in the graph.
        subprocess.run(
            [sys.executable, "-m", "tf2onnx.convert", "--tflite", str(src), "--output", str(dst),
             "--opset", "13", "--inputs-as-nchw", input_name],
            check=True,
        )
        print(f"OK  {src.name} -> {dst}")


if __name__ == "__main__":
    main()
