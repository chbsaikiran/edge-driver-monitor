"""Tunable thresholds. All durations are in seconds so behaviour is independent of frame rate."""

from dataclasses import dataclass


@dataclass
class Config:
    # --- Eyes (drowsiness) ---
    ear_threshold: float = 0.21  # used until calibration finishes
    ear_calib_ratio: float = 0.75  # after calibration: threshold = ratio * driver's open-eye EAR
    ear_threshold_min: float = 0.15
    ear_threshold_max: float = 0.27
    eye_closed_seconds: float = 1.5  # eyes shut this long => DROWSY (microsleep)
    perclos_window_seconds: float = 60.0
    perclos_threshold: float = 0.30  # eyes shut for >30% of the window => FATIGUE

    # --- Mouth (yawning) ---
    mar_threshold: float = 0.60
    yawn_seconds: float = 1.0

    # --- Head pose (distraction), degrees away from the calibrated neutral pose ---
    yaw_limit_deg: float = 25.0
    pitch_limit_deg: float = 20.0
    distraction_seconds: float = 2.0
    no_face_seconds: float = 3.0

    # --- Calibration ---
    calibration_seconds: float = 2.0

    # --- GPIO (BCM numbering), only used with --gpio ---
    buzzer_pin: int = 18
    led_pin: int = 23
