"""Local alert outputs: speaker beep on a laptop, buzzer + LED over GPIO on a Jetson / Raspberry Pi."""

import platform
import shutil
import subprocess
import time
from pathlib import Path

from .monitor import YAWN

# A yawn only lights the LED; everything else also sounds the buzzer.
SILENT_KINDS = {YAWN}

_LINUX_SOUNDS = (
    "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga",
    "/usr/share/sounds/alsa/Front_Center.wav",
)


class SoundAlert:
    """Repeats a system sound while active. Non-blocking: the player runs as a subprocess."""

    def __init__(self, interval: float = 0.6):
        self.interval = interval
        self._cmd = self._find_player()
        self._proc: subprocess.Popen | None = None
        self._last = 0.0

    @staticmethod
    def _find_player() -> list[str] | None:
        if platform.system() == "Darwin" and shutil.which("afplay"):
            return ["afplay", "/System/Library/Sounds/Sosumi.aiff"]
        for sound in _LINUX_SOUNDS:
            if not Path(sound).exists():
                continue
            for player in ("paplay", "aplay"):
                if shutil.which(player) and (player == "paplay" or sound.endswith(".wav")):
                    return [player, sound]
        return None  # fall back to the terminal bell

    def set(self, buzzer: bool, led: bool) -> None:
        if not buzzer:
            return
        now = time.monotonic()
        if now - self._last < self.interval or (self._proc and self._proc.poll() is None):
            return
        self._last = now
        if self._cmd:
            self._proc = subprocess.Popen(
                self._cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            print("\a", end="", flush=True)

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


class GpioAlert:
    """Active-high buzzer and LED. Jetson.GPIO and RPi.GPIO share the same API."""

    def __init__(self, buzzer_pin: int, led_pin: int):
        try:
            import Jetson.GPIO as GPIO
        except ImportError:
            import RPi.GPIO as GPIO
        self._gpio = GPIO
        self._pins = (buzzer_pin, led_pin)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(list(self._pins), GPIO.OUT, initial=GPIO.LOW)

    def set(self, buzzer: bool, led: bool) -> None:
        self._gpio.output(self._pins[0], self._gpio.HIGH if buzzer else self._gpio.LOW)
        self._gpio.output(self._pins[1], self._gpio.HIGH if led else self._gpio.LOW)

    def close(self) -> None:
        self.set(False, False)
        self._gpio.cleanup(list(self._pins))


class Alerter:
    def __init__(self, outputs: list):
        self._outputs = outputs

    def update(self, active: set[str]) -> None:
        led = bool(active)
        buzzer = bool(active - SILENT_KINDS)
        for out in self._outputs:
            out.set(buzzer, led)

    def close(self) -> None:
        for out in self._outputs:
            out.close()
