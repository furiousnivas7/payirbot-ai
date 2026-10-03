"""
Tests for the SCAN PLANT / RESUME control logic (no camera, model or GPIO).

Run:
    python3 -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import db
from inspection import make_command_sender, record_inspection, scan_plant

MIN_CONFIDENCE = 0.70


class FakeRobot:
    """Records GPIO commands instead of touching pins."""

    def __init__(self):
        self.calls = []

    def stop(self):
        self.calls.append("stop")

    def forward(self):
        self.calls.append("forward")


class ScanFlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        # Point the real db module at a temporary database + image folder
        self.images_dir = os.path.join(self.tmp.name, "images")
        os.makedirs(self.images_dir)
        db.DB_PATH = os.path.join(self.tmp.name, "test.db")
        db.IMAGES_DIR = self.images_dir
        db.init_db()

        self.robot = FakeRobot()
        self.send = make_command_sender(lambda: self.robot)

    def inspect_with(self, predict_fn):
        def inspect(image):
            result = record_inspection(
                image,
                predict_fn,
                MIN_CONFIDENCE,
                self.images_dir,
                db.get_next_plant_number,
                db.save_inspection,
            )
            return result.accepted

        return inspect

    def capture(self):
        return Image.new("RGB", (64, 64), "green")

    def saved_count(self):
        return len(db.get_all_inspections())

    def saved_images(self):
        return os.listdir(self.images_dir)

    # 1. good prediction -> saved, robot moves on
    def test_good_prediction_saves_and_resumes(self):
        resumed = scan_plant(
            self.send, self.capture,
            self.inspect_with(lambda img: ("Tomato___Early_blight", 0.82)),
        )
        self.assertTrue(resumed)
        self.assertEqual(self.robot.calls, ["stop", "forward"])
        self.assertEqual(self.saved_count(), 1)
        self.assertEqual(self.saved_images(), ["plant_1.jpg"])
        self.assertEqual(db.get_all_inspections()[0][1], "Tomato - Early blight")

    # 2. low confidence -> nothing saved, robot stays stopped
    def test_low_confidence_saves_nothing_and_stays_stopped(self):
        resumed = scan_plant(
            self.send, self.capture,
            self.inspect_with(lambda img: ("Tomato___Early_blight", 0.54)),
        )
        self.assertFalse(resumed)
        self.assertEqual(self.robot.calls, ["stop"])
        self.assertEqual(self.saved_count(), 0)
        self.assertEqual(self.saved_images(), [])

    # exactly at the threshold counts as accepted
    def test_threshold_boundary_is_accepted(self):
        resumed = scan_plant(
            self.send, self.capture,
            self.inspect_with(lambda img: ("Apple___healthy", MIN_CONFIDENCE)),
        )
        self.assertTrue(resumed)
        self.assertEqual(self.saved_count(), 1)

    # 3. camera failure -> robot stays stopped
    def test_camera_failure_stays_stopped(self):
        errors = []

        def broken_camera():
            raise RuntimeError("Could not open USB webcam.")

        resumed = scan_plant(
            self.send, broken_camera,
            self.inspect_with(lambda img: ("Apple___healthy", 0.99)),
            on_error=errors.append,
        )
        self.assertFalse(resumed)
        self.assertEqual(self.robot.calls, ["stop"])
        self.assertEqual(self.saved_count(), 0)
        self.assertEqual(len(errors), 1)

    # 4. model failure -> robot stays stopped, nothing saved
    def test_model_failure_stays_stopped(self):
        errors = []

        def broken_model(img):
            raise RuntimeError("interpreter failed")

        resumed = scan_plant(
            self.send, self.capture, self.inspect_with(broken_model),
            on_error=errors.append,
        )
        self.assertFalse(resumed)
        self.assertEqual(self.robot.calls, ["stop"])
        self.assertEqual(self.saved_count(), 0)
        self.assertEqual(len(errors), 1)

    # RESUME ROBOT button after a stopped scan -> one FORWARD pulse
    def test_manual_resume_sends_forward_after_failures(self):
        scan_plant(
            self.send, self.capture,
            self.inspect_with(lambda img: ("Tomato___Early_blight", 0.30)),
        )
        self.assertEqual(self.robot.calls, ["stop"])

        self.assertTrue(self.send("RESUME"))
        self.assertEqual(self.robot.calls, ["stop", "forward"])

    # buttons with no GPIO (e.g. on a Mac)
    def test_commands_report_failure_without_robot(self):
        send = make_command_sender(lambda: None)
        self.assertFalse(send("STOP"))
        self.assertFalse(send("RESUME"))

    def test_unknown_command_is_rejected(self):
        self.assertFalse(self.send("REVERSE"))
        self.assertEqual(self.robot.calls, [])


if __name__ == "__main__":
    unittest.main()
