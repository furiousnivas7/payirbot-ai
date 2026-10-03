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

Only ONE Python process may own these pins. Do not run the dashboard and
robot_keyboard.py at the same time.

Safety rules enforced here:
    - every output starts LOW and is set LOW on exit
    - FORWARD and REVERSE are never HIGH together
    - SERVO CW and SERVO CCW are never HIGH together
    - ESP32 inputs should use INPUT_PULLDOWN (the Pi releases the pins on exit)
"""

import atexit
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
