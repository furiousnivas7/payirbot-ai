"""
PayirBot 2.0 - Inspection logic (no Streamlit, no hardware imports)

Decision rules used by dashboard.py, kept here so they can be tested
without a camera, a model or GPIO.
"""

import os
from dataclasses import dataclass
from typing import Optional


def format_disease_name(raw_label: str) -> str:
    """Tomato___Early_blight -> Tomato - Early blight"""
    parts = raw_label.split("___")
    if len(parts) == 2:
        crop, disease = parts
        return f"{crop} - {disease.replace('_', ' ')}"
    return raw_label.replace("_", " ")


@dataclass
class InspectionResult:
    label: str
    confidence: float
    accepted: bool
    disease_name: Optional[str] = None
    is_healthy: bool = False
    plant_number: Optional[int] = None


def record_inspection(
    image,
    predict_fn,
    min_confidence,
    images_dir,
    get_next_plant_number,
    save_inspection,
) -> InspectionResult:
    """
    Run the model on `image`. If confidence >= min_confidence, save the image
    and a database record and return an accepted result; otherwise save
    nothing and return accepted=False.
    """
    label, confidence = predict_fn(image)

    if confidence < min_confidence:
        return InspectionResult(label, confidence, accepted=False)

    disease_name = format_disease_name(label)
    plant_number = get_next_plant_number()

    image_path = os.path.join(images_dir, f"plant_{plant_number}.jpg")
    image.convert("RGB").save(image_path)
    save_inspection(plant_number, disease_name, confidence, image_path)

    return InspectionResult(
        label,
        confidence,
        accepted=True,
        disease_name=disease_name,
        is_healthy="healthy" in label.lower(),
        plant_number=plant_number,
    )


def scan_plant(send_command, capture, inspect, on_error=None) -> bool:
    """
    STOP -> capture -> inspect. Sends RESUME only if the inspection was
    accepted. Any exception (camera, model, ...) leaves the robot stopped.
    Returns True if the robot was resumed.
    """
    send_command("STOP")

    accepted = False
    try:
        accepted = bool(inspect(capture()))
    except Exception as error:
        if on_error is not None:
            on_error(error)

    if accepted:
        send_command("RESUME")
    return accepted


def make_command_sender(get_robot):
    """
    Build send(command) for the dashboard buttons.
    STOP -> robot.stop() (GPIO22 pulse); RESUME -> robot.forward() (GPIO17 pulse).
    Returns False if the robot/GPIO is unavailable or the command fails.
    """

    def send(command: str) -> bool:
        robot = get_robot()
        if robot is None:
            return False
        try:
            if command == "STOP":
                robot.stop()
            elif command == "RESUME":
                robot.forward()
            else:
                return False
            return True
        except Exception:
            return False

    return send
