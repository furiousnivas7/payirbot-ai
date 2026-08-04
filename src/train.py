"""
PayirBot 2.0 - Training Script
Runs two-stage MobileNetV2 transfer learning:
    Stage 1: train the new head only (base frozen)
    Stage 2: fine-tune the top layers of the base at a low learning rate

Saves:
    - Best checkpoint during training -> models/checkpoints/
    - Final trained model            -> models/final_model.h5
    - Training curves (accuracy/loss)-> results/training_history.png

Run:
    python src/train.py
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    STAGE1_EPOCHS,
    STAGE1_LEARNING_RATE,
    STAGE2_EPOCHS,
    STAGE2_LEARNING_RATE,
    FINE_TUNE_AT_LAYER,
    CHECKPOINTS_DIR,
    FINAL_MODEL_PATH,
    RESULTS_DIR,
)
from data_loader import get_datasets
from build_model import build_model, set_fine_tune_layers


def get_callbacks(checkpoint_name):
    """Checkpointing + early stopping, reused for both training stages."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]


def merge_histories(hist1, hist2):
    """Combine Stage 1 + Stage 2 keras History objects into one dict."""
    merged = {}
    for key in hist1.history:
        merged[key] = hist1.history[key] + hist2.history.get(key, [])
    return merged


def plot_history(history_dict):
    """Save accuracy/loss curves to results/training_history.png."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history_dict["accuracy"], label="Train Accuracy")
    axes[0].plot(history_dict["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_dict["loss"], label="Train Loss")
    axes[1].plot(history_dict["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def main():
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print(f"Loaded {len(class_names)} classes.\n")

    print("Building model...")
    model, base_model = build_model()

    # ── Stage 1: train the new head, base frozen ──
    print("\n" + "=" * 50)
    print("STAGE 1: Training classification head (base frozen)")
    print("=" * 50)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=get_callbacks("stage1_best.h5"),
    )

    # ── Stage 2: unfreeze top layers, fine-tune at low LR ──
    print("\n" + "=" * 50)
    print(f"STAGE 2: Fine-tuning from layer {FINE_TUNE_AT_LAYER} onward")
    print("=" * 50)

    set_fine_tune_layers(base_model, FINE_TUNE_AT_LAYER)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        callbacks=get_callbacks("stage2_best.h5"),
    )

    # ── Save final model + training curves ──
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
    model.save(FINAL_MODEL_PATH)
    print(f"\nFinal model saved to {FINAL_MODEL_PATH}")

    merged_history = merge_histories(history_stage1, history_stage2)
    plot_history(merged_history)

    # ── Quick test set evaluation ──
    print("\nEvaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print("\nFor detailed metrics (precision/recall/F1/confusion matrix), run: python src/evaluate.py")


if __name__ == "__main__":
    main()