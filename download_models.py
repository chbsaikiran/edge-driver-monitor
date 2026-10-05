"""Fetch the face landmarker model into ./models so the system can run fully offline."""

from dms.face import ensure_model

if __name__ == "__main__":
    print(f"Model ready: {ensure_model()}")
