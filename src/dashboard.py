"""
PayirBot 2.0 - Streamlit Dashboard (Raspberry Pi)

Pi:    USB webcam -> TFLite model -> SQLite -> this dashboard (phone/browser)
ESP32: receives STOP / FORWARD pulses from the Pi's GPIO pins (robot_gpio.py).

Images can also be uploaded or fetched from a URL for testing without a camera.
Uses tflite-runtime when installed (Pi); falls back to TensorFlow (Mac).

Run:
    streamlit run src/dashboard.py --server.address 0.0.0.0
"""

import json
import os
import time
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        Interpreter = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from robot_gpio import RobotGPIO
except Exception:   # gpiozero missing (e.g. on a Mac)
    RobotGPIO = None

from config import (
    TFLITE_MODEL_PATH,
    CLASS_NAMES,
    IMG_SIZE,
    RESULTS_DIR,
)

from db import (
    init_db,
    get_next_plant_number,
    save_inspection,
    get_all_inspections,
    get_summary_stats,
    IMAGES_DIR,
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "class_metrics.json")

CAMERA_INDEX = 0
CAMERA_WARMUP_FRAMES = 5   # first webcam frames are often dark/blurry


st.set_page_config(page_title="PayirBot 2.0 Dashboard", page_icon="🌱", layout="wide")

init_db()

if Interpreter is None:
    st.error(
        "No TFLite runtime found. On the Pi run: pip install tflite-runtime"
    )
    st.stop()


# ============================================================
# MODEL + METRICS
# ============================================================

@st.cache_resource
def load_interpreter():
    """Load the TensorFlow Lite model once and cache it."""
    interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource
def load_class_metrics():
    """Per-class precision/recall/F1 saved by evaluate.py."""
    if os.path.exists(CLASS_METRICS_JSON_PATH):
        with open(CLASS_METRICS_JSON_PATH, "r") as f:
            return json.load(f)
    return None


def get_trust_level(precision: float) -> tuple:
    """Trust label + Streamlit alert function from historical class precision."""
    if precision >= 0.95:
        return "High trust", st.success
    elif precision >= 0.85:
        return "Moderate trust", st.warning
    return "Low trust - verify manually", st.error


# ============================================================
# ESP32 (GPIO)
# ============================================================

@st.cache_resource
def get_robot():
    """Create the GPIO controller once (it owns the pins); None if unavailable."""
    if RobotGPIO is None:
        return None
    try:
        return RobotGPIO()
    except Exception:
        return None


def send_esp32_command(command: str) -> bool:
    """
    STOP   -> 100 ms pulse on GPIO22
    RESUME -> 100 ms FORWARD pulse on GPIO17 (the ESP32 keeps the motion state)
    Returns False if GPIO is not available.
    """
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


# ============================================================
# IMAGE SOURCES
# ============================================================

def capture_from_camera() -> Image.Image:
    """Grab one frame from the USB webcam as an RGB PIL image."""
    if cv2 is None:
        raise RuntimeError("opencv is not installed (pip install opencv-python).")

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open USB webcam.")

    try:
        frame = None
        for _ in range(CAMERA_WARMUP_FRAMES):
            ok, frame = camera.read()
            if not ok:
                frame = None
            time.sleep(0.1)
    finally:
        camera.release()

    if frame is None:
        raise RuntimeError("Could not capture image from webcam.")

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def load_image_from_url(url: str) -> Image.Image:
    """Download a direct image URL (JPG/PNG/WEBP) as a PIL image."""
    url = url.strip()
    if not url:
        raise ValueError("Please enter an image URL.")

    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if not content_type.startswith("image/"):
        raise ValueError(
            "The URL did not return an image. "
            f"Server returned: {content_type or 'unknown content type'}. "
            "Please use a direct image URL."
        )

    try:
        image = Image.open(BytesIO(response.content))
        image.load()
        return image
    except Exception as e:
        raise ValueError(
            "The URL returned data, but it could not be read as a valid image."
        ) from e


# ============================================================
# PREDICTION
# ============================================================

def predict(interpreter, image: Image.Image):
    """
    Preprocess a PIL image the same way as training (MobileNetV2: [-1, 1])
    and run TFLite inference. Returns (label, confidence).
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    img = np.array(image.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    img = img / 127.5 - 1.0

    # Quantized-input models expect integers
    if input_details["dtype"] != np.float32:
        scale, zero_point = input_details["quantization"]
        info = np.iinfo(input_details["dtype"])
        img = np.clip(np.round(img / scale + zero_point), info.min, info.max)
        img = img.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], np.expand_dims(img, axis=0))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    if output_details["dtype"] != np.float32:
        scale, zero_point = output_details["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    index = int(np.argmax(output))
    return CLASS_NAMES[index], float(output[index])


def format_disease_name(raw_label: str) -> str:
    """Tomato___Early_blight -> Tomato - Early blight"""
    parts = raw_label.split("___")
    if len(parts) == 2:
        crop, disease = parts
        return f"{crop} - {disease.replace('_', ' ')}"
    return raw_label.replace("_", " ")


# ============================================================
# INSPECTION
# ============================================================

def run_inspection(image: Image.Image):
    """Predict, save image + DB record, and render the result."""
    with st.spinner("Analyzing..."):
        label, confidence = predict(load_interpreter(), image)

        disease_name = format_disease_name(label)
        is_healthy = "healthy" in label.lower()
        plant_number = get_next_plant_number()

        image_save_path = os.path.join(IMAGES_DIR, f"plant_{plant_number}.jpg")
        image.convert("RGB").save(image_save_path)

        save_inspection(plant_number, disease_name, confidence, image_save_path)

    st.success("Inspection complete!")

    if is_healthy:
        st.markdown(f"### ✅ Plant {plant_number}: **{disease_name}**")
    else:
        st.markdown(f"### ⚠️ Plant {plant_number}: **{disease_name}**")
        st.warning("Status: Needs Treatment")

    st.metric("Confidence", f"{confidence * 100:.1f}%")

    class_metrics = load_class_metrics()
    if class_metrics and label in class_metrics:
        m = class_metrics[label]
        precision, recall, f1 = m["precision"], m["recall"], m["f1-score"]
        trust_label, alert_fn = get_trust_level(precision)

        st.divider()
        st.caption(
            "Prediction reliability (based on held-out test set for this exact class)"
        )
        alert_fn(
            f"**{trust_label}** — when the model predicts this class, it's correct "
            f"**{precision * 100:.1f}%** of the time (precision), and it catches "
            f"**{recall * 100:.1f}%** of real cases of this disease (recall)."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{precision * 100:.1f}%")
        m2.metric("Recall", f"{recall * 100:.1f}%")
        m3.metric("F1-score", f"{f1 * 100:.1f}%")
    else:
        st.info(
            "No historical reliability data found for this class. "
            "Run `python src/evaluate.py` to generate it."
        )


# ============================================================
# HEADER + STATUS
# ============================================================

st.title("🌱 PayirBot 2.0 - Crop Inspection Dashboard")

status1, status2, status3 = st.columns(3)
if get_robot() is not None:
    status1.success("🟢 Robot GPIO ready")
else:
    status1.warning("🟡 Robot GPIO unavailable")
status2.success("🟢 TFLite model ready")
if cv2 is not None:
    status3.success("🟢 Camera library ready")
else:
    status3.warning("🟡 opencv not installed")

st.divider()


# ============================================================
# ROBOT CONTROL + SCAN
# ============================================================

st.subheader("🤖 Robot Control")

c_resume, c_stop, c_scan = st.columns(3)

if c_resume.button("▶️ RESUME ROBOT", use_container_width=True):
    if send_esp32_command("RESUME"):
        st.success("Robot moving")
    else:
        st.error("Robot GPIO unavailable")

if c_stop.button("⛔ STOP ROBOT", use_container_width=True):
    if send_esp32_command("STOP"):
        st.warning("Robot stopped")
    else:
        st.error("Robot GPIO unavailable")

if c_scan.button("📷 SCAN PLANT", type="primary", use_container_width=True):
    send_esp32_command("STOP")
    st.info("Robot stopped. Capturing image...")
    try:
        captured = capture_from_camera()
        st.image(captured, caption="Captured plant image")
        run_inspection(captured)
    except Exception as e:
        st.error(f"Inspection failed: {e}")
    finally:
        send_esp32_command("RESUME")
        st.info("Robot resumed.")

st.divider()


# ============================================================
# TEST INSPECTION (upload / URL, no camera needed)
# ============================================================

with st.expander("🧪 Test with an uploaded image or URL"):
    input_method = st.radio(
        "Choose image source", ["Upload Image", "Image URL"], horizontal=True
    )

    test_image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                test_image = Image.open(uploaded_file)
                test_image.load()
            except Exception as e:
                st.error(f"Could not open image: {e}")
    else:
        image_url = st.text_input("Enter image URL", placeholder="https://example.com/leaf.jpg")
        if image_url:
            try:
                test_image = load_image_from_url(image_url)
            except requests.exceptions.RequestException as e:
                st.error(f"Could not download image: {e}")
            except ValueError as e:
                st.error(str(e))

    if test_image is not None:
        st.image(test_image, caption="Image to analyze")
        if st.button("Run Inspection", type="primary"):
            run_inspection(test_image)

st.divider()


# ============================================================
# SUMMARY (after any inspection so counts are current)
# ============================================================

total, healthy, diseased = get_summary_stats()

col1, col2, col3 = st.columns(3)
col1.metric("Total Inspections", total)
col2.metric("Healthy", healthy)
col3.metric("Diseased", diseased)

st.divider()


# ============================================================
# LATEST RESULT + HISTORY
# ============================================================

records = get_all_inspections()

st.subheader("Latest Result")

if records:
    plant_number, disease, confidence, image_path, timestamp = records[0]

    img_col, info_col = st.columns(2)
    with img_col:
        if os.path.exists(image_path):
            st.image(image_path)
    with info_col:
        st.write(f"**Plant {plant_number}**")
        st.write(f"**Disease:** {disease}")
        st.write(f"**Confidence:** {confidence * 100:.1f}%")
        st.write(f"**Time:** {timestamp}")
else:
    st.info("No inspections yet. Press SCAN PLANT to get started.")

st.divider()
st.subheader("Inspection History")

if records:
    for plant_number, disease, confidence, image_path, timestamp in records:
        c1, c2 = st.columns([1, 4])
        with c1:
            if os.path.exists(image_path):
                st.image(image_path, width=100)
        with c2:
            st.write(f"**Plant {plant_number}** - {disease}")
            st.write(f"Confidence: {confidence * 100:.1f}% | {timestamp}")
        st.divider()
else:
    st.info("No inspection history yet.")
