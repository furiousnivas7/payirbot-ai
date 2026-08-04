"""
PayirBot 2.0 - Streamlit Dashboard
Upload a leaf image, run it through the trained TFLite model, and
view inspection results and history - simulating what the robot's
camera + AI pipeline will do once hardware is built.

Run:
    streamlit run src/dashboard.py
"""

import json
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from config import TFLITE_MODEL_PATH, CLASS_NAMES, IMG_SIZE, RESULTS_DIR
from db import (
    init_db,
    get_next_plant_number,
    save_inspection,
    get_all_inspections,
    get_summary_stats,
    IMAGES_DIR,
)

CLASS_METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "class_metrics.json")


@st.cache_resource
def load_class_metrics():
    """Load per-class precision/recall/f1 from evaluate.py's saved JSON, if present."""
    if os.path.exists(CLASS_METRICS_JSON_PATH):
        with open(CLASS_METRICS_JSON_PATH, "r") as f:
            return json.load(f)
    return None


def get_trust_level(precision: float) -> tuple:
    """Returns (label, streamlit_alert_function) based on how trustworthy this class historically is."""
    if precision >= 0.95:
        return "High trust", st.success
    elif precision >= 0.85:
        return "Moderate trust", st.warning
    else:
        return "Low trust - verify manually", st.error

st.set_page_config(page_title="PayirBot 2.0 Dashboard", layout="wide")


# ─────────────────────────────────────────────
# MODEL LOADING (cached so it only loads once)
# ─────────────────────────────────────────────

@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, image: Image.Image):
    """Preprocess a PIL image and run TFLite inference. Returns (label, confidence)."""
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index])
    label = CLASS_NAMES[predicted_index]

    return label, confidence


def format_disease_name(raw_label: str) -> str:
    """Turn 'Tomato___Early_blight' into 'Tomato - Early blight'."""
    parts = raw_label.split("___")
    if len(parts) == 2:
        crop, disease = parts
        disease = disease.replace("_", " ")
        return f"{crop} - {disease}"
    return raw_label.replace("_", " ")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

init_db()
interpreter = load_interpreter()

st.title("🌱 PayirBot 2.0 - Crop Inspection Dashboard")
st.caption("Upload a leaf image to simulate a robot inspection.")

# Summary stats header
total, healthy, diseased = get_summary_stats()
col1, col2, col3 = st.columns(3)
col1.metric("Total Inspections", total)
col2.metric("Healthy", healthy)
col3.metric("Diseased", diseased)

st.divider()

# Upload + predict section
left, right = st.columns([1, 1])

with left:
    st.subheader("New Inspection")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", width='stretch')

        if st.button("Run Inspection", type="primary"):
            with st.spinner("Analyzing..."):
                label, confidence = predict(interpreter, image)
                disease_name = format_disease_name(label)
                is_healthy = "healthy" in label.lower()

                plant_number = get_next_plant_number()
                image_filename = f"plant_{plant_number}.jpg"
                image_save_path = os.path.join(IMAGES_DIR, image_filename)
                image.convert("RGB").save(image_save_path)

                save_inspection(plant_number, disease_name, confidence, image_save_path)

            st.success("Inspection complete!")

            if is_healthy:
                st.markdown(f"### ✅ Plant {plant_number}: **{disease_name}**")
            else:
                st.markdown(f"### ⚠️ Plant {plant_number}: **{disease_name}**")
                st.warning("Status: Needs Treatment")

            st.metric("Confidence", f"{confidence * 100:.1f}%")

            # Trust context: how reliable is the model historically for THIS class,
            # based on the held-out test set metrics (not just this one prediction's
            # confidence score).
            class_metrics = load_class_metrics()
            if class_metrics and label in class_metrics:
                m = class_metrics[label]
                precision = m["precision"]
                recall = m["recall"]
                f1 = m["f1-score"]

                trust_label, alert_fn = get_trust_level(precision)

                st.divider()
                st.caption("Prediction reliability (based on held-out test set for this exact class)")
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

            st.rerun()

with right:
    st.subheader("Latest Result")
    records = get_all_inspections()
    if records:
        latest = records[0]
        plant_number, disease, confidence, image_path, timestamp = latest
        if os.path.exists(image_path):
            st.image(image_path, width='stretch')
        st.write(f"**Plant {plant_number}**")
        st.write(f"**Disease:** {disease}")
        st.write(f"**Confidence:** {confidence * 100:.1f}%")
        st.write(f"**Time:** {timestamp}")
    else:
        st.info("No inspections yet. Upload an image to get started.")

st.divider()

# History table
st.subheader("Inspection History")
records = get_all_inspections()

if records:
    for plant_number, disease, confidence, image_path, timestamp in records:
        with st.container():
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