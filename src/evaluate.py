"""
PayirBot 2.0 - Evaluation Script
Loads the final trained model and computes detailed metrics on the
held-out test set: precision, recall, F1-score (per class + overall),
and a confusion matrix.

Saves:
    - results/confusion_matrix.png
    - results/metrics_report.txt

Run:
    python src/evaluate.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import FINAL_MODEL_PATH, RESULTS_DIR, CLASS_NAMES
from data_loader import get_datasets

CLASS_METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "class_metrics.json")


def get_predictions(model, test_ds):
    """Run inference over the full test set, return true and predicted labels."""
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - PayirBot 2.0 Disease Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")

    return cm


def save_metrics_report(y_true, y_pred, class_names):
    """Compute and save precision/recall/F1 per class + overall summary."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )

    out_path = os.path.join(RESULTS_DIR, "metrics_report.txt")
    with open(out_path, "w") as f:
        f.write("PayirBot 2.0 - Test Set Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    print(f"Saved metrics report to {out_path}")
    print("\n" + report)

    # Also save as JSON so the dashboard can look up per-class precision/
    # recall/f1 at inference time and show trust context alongside predictions
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    with open(CLASS_METRICS_JSON_PATH, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"Saved per-class metrics JSON to {CLASS_METRICS_JSON_PATH}")


def find_most_confused_pairs(cm, class_names, top_n=10):
    """
    Identify the top N most-confused class pairs (excluding the diagonal),
    useful for discussing which diseases look visually similar.
    """
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], class_names[i], class_names[j]))

    confusions.sort(reverse=True)

    print(f"\nTop {top_n} most confused class pairs (true -> predicted):")
    for count, true_class, pred_class in confusions[:top_n]:
        print(f"  {count:4d}x  {true_class}  ->  {pred_class}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading model from {FINAL_MODEL_PATH}...")
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)

    print("Loading test dataset...")
    _, _, test_ds, class_names = get_datasets()

    print("Running predictions on test set...")
    y_true, y_pred = get_predictions(model, test_ds)

    print("\nComputing metrics...")
    save_metrics_report(y_true, y_pred, class_names)
    cm = save_confusion_matrix(y_true, y_pred, class_names)
    find_most_confused_pairs(cm, class_names)


if __name__ == "__main__":
    main()