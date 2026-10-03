"""
PayirBot 2.0 - Robot GPIO control (Raspberry Pi -> ESP32)

The Pi only sends digital signals. The ESP32 drives the motors, runs the
line-following algorithm and moves the servo.

BCM vs physical pin numbers
---------------------------
This file uses BCM (Broadcom) GPIO numbers: "GPIO17" is BCM 17. That is NOT
physical pin 17. Physical pin = position on the 40-pin header.

    Function        BCM GPIO   Physical pin   Signal
    FORWARD         17         11             100 ms pulse
    REVERSE         27         13             100 ms pulse
    STOP            22         15             100 ms pulse
    SERVO CW        23         16             HIGH while active
    SERVO CCW       24         18             HIGH while active
    SERVO 90        25         22             100 ms pulse
    GND             -          6              common ground with ESP32

ESP32 side of each wire: FORWARD->GPIO4, REVERSE->GPIO5, STOP->GPIO13,
SERVO CW->GPIO14, SERVO CCW->GPIO16, SERVO 90->GPIO17.

Use from another program (functions, no keyboard, no object to manage):

    from robot_gpio import send_forward, send_stop, servo_center
    send_forward()

Only ONE Python process may own these pins. Do not run the dashboard and
robot_keyboard.py at the same time.

Safety rules enforced here:
    - every output starts LOW and is set LOW on exit
    - FORWARD and REVERSE are never HIGH together
    - SERVO CW and SERVO CCW are never HIGH together
    - ESP32 inputs should use INPUT_PULLDOWN (the Pi releases the pins on exit)
"""

import atexit
import signal
import sys
import threading
import time

from gpiozero import OutputDevice

# ── BCM pin numbers ──
FORWARD_PIN = 17
REVERSE_PIN = 27
STOP_PIN = 22
SERVO_CW_PIN = 23
SERVO_CCW_PIN = 24
SERVO_90_PIN = 25

PULSE_DURATION = 0.100   # seconds


class RobotGPIO:
    def __init__(self):
        self._lock = threading.Lock()
        self.forward_pin = OutputDevice(FORWARD_PIN, initial_value=False)
        self.reverse_pin = OutputDevice(REVERSE_PIN, initial_value=False)
        self.stop_pin = OutputDevice(STOP_PIN, initial_value=False)
        self.servo_cw_pin = OutputDevice(SERVO_CW_PIN, initial_value=False)
        self.servo_ccw_pin = OutputDevice(SERVO_CCW_PIN, initial_value=False)
        self.servo_90_pin = OutputDevice(SERVO_90_PIN, initial_value=False)
        self._closed = False
        atexit.register(self.close)

    # ── helpers ──

    def _all_pins(self):
        return (
            self.forward_pin, self.reverse_pin, self.stop_pin,
            self.servo_cw_pin, self.servo_ccw_pin, self.servo_90_pin,
        )

    def all_low(self):
        """Force every output LOW."""
        with self._lock:
            for pin in self._all_pins():
                pin.off()

    def _pulse(self, pin):
        """One HIGH pulse then LOW. Holds the lock so pulses never overlap."""
        with self._lock:
            if self._closed:
                return
            # Never leave both direction lines HIGH
            self.forward_pin.off()
            self.reverse_pin.off()
            pin.on()
            try:
                time.sleep(PULSE_DURATION)
            finally:
                pin.off()

    # ── motor commands (pulses) ──

    def forward(self):
        self._pulse(self.forward_pin)

    def reverse(self):
        self._pulse(self.reverse_pin)

    def stop(self):
        self._pulse(self.stop_pin)

    # ── servo commands ──

    def servo_cw_start(self):
        with self._lock:
            if self._closed:
                return
            self.servo_ccw_pin.off()   # turn the opposite direction off first
            self.servo_cw_pin.on()

    def servo_ccw_start(self):
        with self._lock:
            if self._closed:
                return
            self.servo_cw_pin.off()
            self.servo_ccw_pin.on()

    def servo_cw_release(self):
        with self._lock:
            self.servo_cw_pin.off()

    def servo_ccw_release(self):
        with self._lock:
            self.servo_ccw_pin.off()

    def servo_stop(self):
        with self._lock:
            self.servo_cw_pin.off()
            self.servo_ccw_pin.off()

    def servo_90(self):
        # Don't pulse while the servo is being driven continuously
        self.servo_stop()
        self._pulse(self.servo_90_pin)

    # ── cleanup ──

    def close(self):
        """Set everything LOW, then release the pins. Safe to call twice."""
        if self._closed:
            return
        with self._lock:
            for pin in self._all_pins():
                try:
                    pin.off()
                except Exception:
                    pass
            self._closed = True
            for pin in self._all_pins():
                try:
                    pin.close()
                except Exception:
                    pass


# ─────────────────────────────────────────────
# Function API (lazy singleton)
# ─────────────────────────────────────────────
# The pins are claimed on the first call, not on import, so importing this
# module has no side effects. Ctrl+C is handled by atexit/close(); SIGTERM
# (kill, systemd) is handled too, but only when first called from the main
# thread (Python only allows signal handlers there).

_robot = None
_robot_lock = threading.Lock()


def _get_robot():
    global _robot
    with _robot_lock:
        if _robot is None:
            _robot = RobotGPIO()
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        return _robot


def send_forward():
    """100 ms pulse on GPIO17 -> ESP32 switches to FORWARD."""
    _get_robot().forward()


def send_reverse():
    """100 ms pulse on GPIO27 -> ESP32 switches to REVERSE."""
    _get_robot().reverse()


def send_stop():
    """100 ms pulse on GPIO22 -> ESP32 stops both motors."""
    _get_robot().stop()


def servo_clockwise():
    """GPIO23 HIGH until servo_stop() / servo_counter_clockwise()."""
    _get_robot().servo_cw_start()


def servo_counter_clockwise():
    """GPIO24 HIGH until servo_stop() / servo_clockwise()."""
    _get_robot().servo_ccw_start()


def servo_stop():
    """GPIO23 and GPIO24 LOW."""
    _get_robot().servo_stop()


def servo_center():
    """100 ms pulse on GPIO25 -> ESP32 moves the camera servo to 90 degrees."""
    _get_robot().servo_90()


def close():
    """Set all outputs LOW and release the pins (also runs at exit)."""
    if _robot is not None:
        _robot.close()
