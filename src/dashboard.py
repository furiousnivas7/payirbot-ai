"""
PayirBot 2.0 - Streamlit Dashboard

Upload a leaf image OR provide an image URL, run it through the
trained TFLite model, and view inspection results and history.

Run:
    streamlit run src/dashboard.py
"""

import json
import os
from io import BytesIO

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image

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

CLASS_METRICS_JSON_PATH = os.path.join(
    RESULTS_DIR,
    "class_metrics.json"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PayirBot 2.0 Dashboard",
    layout="wide"
)


# ============================================================
# LOAD CLASS METRICS
# ============================================================

@st.cache_resource
def load_class_metrics():
    """
    Load per-class precision, recall and F1-score
    from evaluate.py's saved JSON file.
    """

    if os.path.exists(CLASS_METRICS_JSON_PATH):

        with open(
            CLASS_METRICS_JSON_PATH,
            "r"
        ) as f:

            return json.load(f)

    return None


# ============================================================
# TRUST LEVEL
# ============================================================

def get_trust_level(precision: float) -> tuple:
    """
    Return a trust label and Streamlit alert function
    based on historical class precision.
    """

    if precision >= 0.95:
        return "High trust", st.success

    elif precision >= 0.85:
        return "Moderate trust", st.warning

    else:
        return "Low trust - verify manually", st.error


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

@st.cache_resource
def load_interpreter():
    """
    Load the TensorFlow Lite model once and cache it.
    """

    interpreter = tf.lite.Interpreter(
        model_path=TFLITE_MODEL_PATH
    )

    interpreter.allocate_tensors()

    return interpreter


# ============================================================
# LOAD IMAGE FROM URL
# ============================================================

def load_image_from_url(url: str):
    """
    Download an image from a URL and return it as a PIL Image.

    The URL must return an actual image such as:
        JPG
        JPEG
        PNG
        WEBP

    It should not be a normal webpage URL.
    """

    url = url.strip()

    if not url:
        raise ValueError(
            "Please enter an image URL."
        )

    response = requests.get(
        url,
        timeout=15,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    # Check server response type
    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if not content_type.startswith("image/"):

        raise ValueError(
            "The URL did not return an image. "
            f"Server returned: "
            f"{content_type or 'unknown content type'}. "
            "Please use a direct image URL."
        )

    try:

        image = Image.open(
            BytesIO(response.content)
        )

        # Force PIL to load the image completely
        image.load()

        return image

    except Exception as e:

        raise ValueError(
            "The URL returned data, but it could not "
            "be read as a valid image. "
            "Please use a direct JPG, JPEG, PNG, "
            "or WEBP image URL."
        ) from e


# ============================================================
# PREDICTION
# ============================================================

def predict(
    interpreter,
    image: Image.Image
):
    """
    Preprocess a PIL image and run TFLite inference.

    Returns:
        label, confidence
    """

    # --------------------------------------------------------
    # Convert image to RGB and resize
    # --------------------------------------------------------

    img = image.convert(
        "RGB"
    ).resize(
        IMG_SIZE
    )

    # --------------------------------------------------------
    # Convert to NumPy array
    # --------------------------------------------------------

    img_array = np.array(
        img,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # MobileNetV2 preprocessing
    # --------------------------------------------------------

    img_array = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            img_array
        )
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # --------------------------------------------------------
    # Get model input/output details
    # --------------------------------------------------------

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    # --------------------------------------------------------
    # Send image to TFLite model
    # --------------------------------------------------------

    interpreter.set_tensor(
        input_details[0]["index"],
        img_array
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    interpreter.invoke()

    # --------------------------------------------------------
    # Get prediction output
    # --------------------------------------------------------

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # --------------------------------------------------------
    # Find highest probability
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(output)
    )

    confidence = float(
        output[predicted_index]
    )

    label = CLASS_NAMES[
        predicted_index
    ]

    return label, confidence


# ============================================================
# FORMAT DISEASE NAME
# ============================================================

def format_disease_name(
    raw_label: str
) -> str:
    """
    Convert:

        Tomato___Early_blight

    into:

        Tomato - Early blight
    """

    parts = raw_label.split(
        "___"
    )

    if len(parts) == 2:

        crop, disease = parts

        disease = disease.replace(
            "_",
            " "
        )

        return f"{crop} - {disease}"

    return raw_label.replace(
        "_",
        " "
    )


# ============================================================
# INITIALIZE DATABASE + MODEL
# ============================================================

init_db()

interpreter = load_interpreter()


# ============================================================
# APP HEADER
# ============================================================

st.title(
    "🌱 PayirBot 2.0 - Crop Inspection Dashboard"
)

st.caption(
    "Upload a leaf image or provide an image URL "
    "to simulate a robot inspection."
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

total, healthy, diseased = (
    get_summary_stats()
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Inspections",
    total
)

col2.metric(
    "Healthy",
    healthy
)

col3.metric(
    "Diseased",
    diseased
)


st.divider()


# ============================================================
# NEW INSPECTION + LATEST RESULT
# ============================================================

left, right = st.columns(
    [1, 1]
)


# ============================================================
# LEFT COLUMN - NEW INSPECTION
# ============================================================

with left:

    st.subheader(
        "New Inspection"
    )

    # --------------------------------------------------------
    # Choose image source
    # --------------------------------------------------------

    input_method = st.radio(
        "Choose image source",
        [
            "Upload Image",
            "Image URL"
        ],
        horizontal=True
    )

    image = None


    # ========================================================
    # OPTION 1 - UPLOAD IMAGE
    # ========================================================

    if input_method == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a leaf image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

        if uploaded_file is not None:

            try:

                image = Image.open(
                    uploaded_file
                )

                image.load()

            except Exception as e:

                st.error(
                    f"Could not open image: {e}"
                )


    # ========================================================
    # OPTION 2 - IMAGE URL
    # ========================================================

    else:

        image_url = st.text_input(
            "Enter image URL",
            placeholder=(
                "https://example.com/leaf.jpg"
            )
        )

        if image_url:

            try:

                image = load_image_from_url(
                    image_url
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The image request timed out. "
                    "Please try another URL."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the image URL. "
                    "Please check your internet connection "
                    "and the URL."
                )

            except requests.exceptions.HTTPError as e:

                st.error(
                    f"The image server returned an error: {e}"
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Could not download image: {e}"
                )

            except ValueError as e:

                st.error(
                    str(e)
                )

            except Exception as e:

                st.error(
                    f"Could not load image: {e}"
                )


    # ========================================================
    # SHOW IMAGE
    # ========================================================

    if image is not None:

        st.image(
            image,
            caption="Image to analyze"
        )


        # ====================================================
        # RUN INSPECTION BUTTON
        # ====================================================

        if st.button(
            "Run Inspection",
            type="primary"
        ):

            with st.spinner(
                "Analyzing..."
            ):

                # ------------------------------------------------
                # Run model prediction
                # ------------------------------------------------

                label, confidence = predict(
                    interpreter,
                    image
                )


                # ------------------------------------------------
                # Format prediction
                # ------------------------------------------------

                disease_name = (
                    format_disease_name(
                        label
                    )
                )

                is_healthy = (
                    "healthy"
                    in label.lower()
                )


                # ------------------------------------------------
                # Create plant number
                # ------------------------------------------------

                plant_number = (
                    get_next_plant_number()
                )


                # ------------------------------------------------
                # Save image
                # ------------------------------------------------

                image_filename = (
                    f"plant_{plant_number}.jpg"
                )

                image_save_path = os.path.join(
                    IMAGES_DIR,
                    image_filename
                )


                image.convert(
                    "RGB"
                ).save(
                    image_save_path
                )


                # ------------------------------------------------
                # Save inspection to database
                # ------------------------------------------------

                save_inspection(
                    plant_number,
                    disease_name,
                    confidence,
                    image_save_path
                )


            # ====================================================
            # INSPECTION COMPLETE
            # ====================================================

            st.success(
                "Inspection complete!"
            )


            # ====================================================
            # PREDICTION RESULT
            # ====================================================

            if is_healthy:

                st.markdown(
                    f"### ✅ Plant {plant_number}: "
                    f"**{disease_name}**"
                )

            else:

                st.markdown(
                    f"### ⚠️ Plant {plant_number}: "
                    f"**{disease_name}**"
                )

                st.warning(
                    "Status: Needs Treatment"
                )


            # ====================================================
            # CONFIDENCE
            # ====================================================

            st.metric(
                "Confidence",
                f"{confidence * 100:.1f}%"
            )


            # ====================================================
            # CLASS RELIABILITY
            # ====================================================

            class_metrics = (
                load_class_metrics()
            )


            if (
                class_metrics
                and label in class_metrics
            ):

                m = class_metrics[label]

                precision = m[
                    "precision"
                ]

                recall = m[
                    "recall"
                ]

                f1 = m[
                    "f1-score"
                ]


                trust_label, alert_fn = (
                    get_trust_level(
                        precision
                    )
                )


                st.divider()


                st.caption(
                    "Prediction reliability "
                    "(based on held-out test "
                    "set for this exact class)"
                )


                alert_fn(
                    f"**{trust_label}** — when the "
                    f"model predicts this class, "
                    f"it's correct "
                    f"**{precision * 100:.1f}%** "
                    f"of the time (precision), "
                    f"and it catches "
                    f"**{recall * 100:.1f}%** "
                    f"of real cases of this "
                    f"disease (recall)."
                )


                m1, m2, m3 = st.columns(
                    3
                )


                m1.metric(
                    "Precision",
                    f"{precision * 100:.1f}%"
                )


                m2.metric(
                    "Recall",
                    f"{recall * 100:.1f}%"
                )


                m3.metric(
                    "F1-score",
                    f"{f1 * 100:.1f}%"
                )


            else:

                st.info(
                    "No historical reliability "
                    "data found for this class. "
                    "Run `python src/evaluate.py` "
                    "to generate it."
                )


            # ----------------------------------------------------
            # Refresh dashboard
            # ----------------------------------------------------

            st.experimental_rerun()


# ============================================================
# RIGHT COLUMN - LATEST RESULT
# ============================================================

with right:

    st.subheader(
        "Latest Result"
    )

    records = (
        get_all_inspections()
    )


    if records:

        latest = records[0]

        (
            plant_number,
            disease,
            confidence,
            image_path,
            timestamp
        ) = latest


        if os.path.exists(
            image_path
        ):

            st.image(
                image_path
            )


        st.write(
            f"**Plant {plant_number}**"
        )

        st.write(
            f"**Disease:** {disease}"
        )

        st.write(
            f"**Confidence:** "
            f"{confidence * 100:.1f}%"
        )

        st.write(
            f"**Time:** {timestamp}"
        )


    else:

        st.info(
            "No inspections yet. "
            "Upload an image to get started."
        )


# ============================================================
# INSPECTION HISTORY
# ============================================================

st.divider()

st.subheader(
    "Inspection History"
)

records = (
    get_all_inspections()
)


if records:

    for (
        plant_number,
        disease,
        confidence,
        image_path,
        timestamp
    ) in records:

        with st.container():

            c1, c2 = st.columns(
                [1, 4]
            )


            # ------------------------------------------------
            # HISTORY IMAGE
            # ------------------------------------------------

            with c1:

                if os.path.exists(
                    image_path
                ):

                    st.image(
                        image_path,
                        width=100
                    )


            # ------------------------------------------------
            # HISTORY DETAILS
            # ------------------------------------------------

            with c2:

                st.write(
                    f"**Plant {plant_number}** "
                    f"- {disease}"
                )

                st.write(
                    f"Confidence: "
                    f"{confidence * 100:.1f}% "
                    f"| {timestamp}"
                )


            st.divider()


else:

    st.info(
        "No inspection history yet."
    )

