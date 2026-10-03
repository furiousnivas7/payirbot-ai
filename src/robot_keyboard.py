"""
PayirBot 2.0 - Keyboard test controller (Raspberry Pi, evdev)

Reads key press AND release events straight from /dev/input, so it works on
Wayland and over SSH. The Pi user must be in the "input" group (default on
Raspberry Pi OS; check with: groups).

    W  Forward (100 ms pulse)        A  Servo clockwise (while held)
    S  Reverse (100 ms pulse)        D  Servo counter-clockwise (while held)
    X  Stop (100 ms pulse)           Q  Stop servo
    C  Servo to 90 degrees (pulse)   ESC  Exit

Run (do NOT run while the dashboard is running - one process owns the GPIO):
    python3 src/robot_keyboard.py
"""

import signal
import sys

from evdev import InputDevice, ecodes, list_devices

from robot_gpio import RobotGPIO

KEY_PRESS = 1
KEY_RELEASE = 0
# value 2 = auto-repeat while held; ignored on purpose


def find_keyboard():
    """Return the first input device that looks like a keyboard."""
    for path in list_devices():
        device = InputDevice(path)
        keys = device.capabilities().get(ecodes.EV_KEY, [])
        if ecodes.KEY_W in keys and ecodes.KEY_ESC in keys:
            return device
    return None


def main():
    keyboard = find_keyboard()
    if keyboard is None:
        sys.exit("No keyboard found. Is your user in the 'input' group?")

    robot = RobotGPIO()   # all outputs start LOW

    # Make Ctrl+C and `kill` clean up too
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # key code -> action on press
    pulse_keys = {
        ecodes.KEY_W: ("FORWARD", robot.forward),
        ecodes.KEY_S: ("REVERSE", robot.reverse),
        ecodes.KEY_X: ("STOP", robot.stop),
        ecodes.KEY_C: ("SERVO 90", robot.servo_90),
        ecodes.KEY_Q: ("SERVO STOP", robot.servo_stop),
    }

    print(f"Using keyboard: {keyboard.name}")
    print("W fwd | S rev | X stop | A/D servo (hold) | Q servo stop | C servo 90 | ESC exit")

    try:
        for event in keyboard.read_loop():
            if event.type != ecodes.EV_KEY:
                continue

            if event.value == KEY_PRESS:
                if event.code == ecodes.KEY_ESC:
                    break
                elif event.code == ecodes.KEY_A:
                    print("SERVO CLOCKWISE")
                    robot.servo_cw_start()
                elif event.code == ecodes.KEY_D:
                    print("SERVO COUNTER-CLOCKWISE")
                    robot.servo_ccw_start()
                elif event.code in pulse_keys:
                    name, action = pulse_keys[event.code]
                    print(name)
                    action()

            elif event.value == KEY_RELEASE:
                if event.code == ecodes.KEY_A:
                    robot.servo_cw_release()
                    print("SERVO STOP")
                elif event.code == ecodes.KEY_D:
                    robot.servo_ccw_release()
                    print("SERVO STOP")

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("Setting all GPIO outputs LOW...")
        robot.close()
        print("Controller stopped safely.")


if __name__ == "__main__":
    main()
