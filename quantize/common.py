"""Paths and model facts shared by the conversion / quantization scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASK_FILE = ROOT.parent / "models" / "face_landmarker.task"
WORK = ROOT / "work"
ORIGINAL = WORK / "original"  # unpacked .task bundle
ONNX = WORK / "onnx"
SAVED_MODEL = WORK / "saved_model"
INT8 = WORK / "int8"
CALIB = ROOT / "calib"

# name -> (input height/width, value range MediaPipe feeds the model)
MODELS = {
    "face_detector": (128, (-1.0, 1.0)),
    "face_landmarks_detector": (256, (0.0, 1.0)),
}
