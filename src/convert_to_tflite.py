"""
PayirBot 2.0 - TensorFlow Lite Conversion
Converts the trained Keras model (models/final_model.h5) into a
quantized TensorFlow Lite model suitable for the Raspberry Pi Zero 2W.

Also runs a quick sanity check comparing the original Keras model's
accuracy against the converted TFLite model's accuracy on a sample
of the test set, so you can confirm quantization didn't hurt accuracy
much before deploying.

Saves:
    - models/payirbot_model.tflite

Run:
    python src/convert_to_tflite.py
"""

import os
import shutil
import time
import numpy as np
import tensorflow as tf

from config import FINAL_MODEL_PATH, TFLITE_MODEL_PATH, MODELS_DIR, CLASS_NAMES
from data_loader import get_datasets

SAVEDMODEL_EXPORT_DIR = os.path.join(MODELS_DIR, "final_model_savedmodel")


def convert_to_tflite():
    """
    Load the Keras model, export it to a clean SavedModel first (this
    avoids a known TFLite converter crash - "missing attribute 'value'"
    in MLIR's variable-freezing pass - that happens when converting
    directly from an H5-loaded model with a nested sub-model like
    MobileNetV2), then convert from the SavedModel to TFLite.
    """
    print(f"Loading Keras model from {FINAL_MODEL_PATH}...")
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)

    # Clean re-export as SavedModel - this resolves the nested variable
    # freezing issue that crashes the converter when going straight
    # from an H5-loaded model.
    if os.path.exists(SAVEDMODEL_EXPORT_DIR):
        shutil.rmtree(SAVEDMODEL_EXPORT_DIR)

    print(f"Exporting clean SavedModel to {SAVEDMODEL_EXPORT_DIR}...")
    model.export(SAVEDMODEL_EXPORT_DIR)

    print("Converting SavedModel to TensorFlow Lite (with dynamic range quantization)...")
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_EXPORT_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantizes weights to int8 where possible
    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Saved TFLite model to {TFLITE_MODEL_PATH} ({size_mb:.2f} MB)")

    return model


def evaluate_tflite_vs_keras(keras_model, num_samples=300):
    """
    Runs a subset of the test set through both the original Keras model
    and the converted TFLite model, comparing accuracy and per-image
    inference time. This is NOT a full test-set re-evaluation (that's
    already done in evaluate.py) - it's a quick quantization sanity check.
    """
    print(f"\nRunning comparison on {num_samples} test images...")

    _, _, test_ds, class_names = get_datasets()

    # Set up the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    keras_correct = 0
    tflite_correct = 0
    total = 0
    tflite_times = []

    for images, labels in test_ds:
        for i in range(images.shape[0]):
            if total >= num_samples:
                break

            image = images[i : i + 1]
            true_label = np.argmax(labels[i].numpy())

            # Keras prediction
            keras_pred = np.argmax(keras_model.predict(image, verbose=0)[0])
            if keras_pred == true_label:
                keras_correct += 1

            # TFLite prediction
            interpreter.set_tensor(input_details[0]["index"], image)
            start = time.time()
            interpreter.invoke()
            tflite_times.append(time.time() - start)
            tflite_output = interpreter.get_tensor(output_details[0]["index"])
            tflite_pred = np.argmax(tflite_output[0])
            if tflite_pred == true_label:
                tflite_correct += 1

            total += 1

        if total >= num_samples:
            break

    avg_tflite_time_ms = (sum(tflite_times) / len(tflite_times)) * 1000

    print("\n" + "=" * 50)
    print("QUANTIZATION SANITY CHECK")
    print("=" * 50)
    print(f"Samples evaluated:       {total}")
    print(f"Keras model accuracy:    {keras_correct / total:.4f}")
    print(f"TFLite model accuracy:   {tflite_correct / total:.4f}")
    print(f"Avg TFLite inference:    {avg_tflite_time_ms:.2f} ms/image (on this Mac, not the Pi)")
    print("=" * 50)
    print(
        "\nNote: inference time on the Raspberry Pi Zero 2W will be slower "
        "than on this Mac - test this once deployed to confirm it meets "
        "your target inspection pace."
    )


if __name__ == "__main__":
    keras_model = convert_to_tflite()
    evaluate_tflite_vs_keras(keras_model, num_samples=300)