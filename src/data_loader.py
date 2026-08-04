"""
PayirBot 2.0 - Data Loading
Builds tf.data pipelines for train/val/test from data/processed/.

Used by train.py and evaluate.py - not meant to be run directly.
"""

import tensorflow as tf

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
)

AUTOTUNE = tf.data.AUTOTUNE


# ─────────────────────────────────────────────
# AUGMENTATION (training set only)
# ─────────────────────────────────────────────

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ],
    name="data_augmentation",
)


def _load_dataset_from_directory(directory, shuffle):
    """Load images from a class-labeled folder into a tf.data.Dataset."""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",     # one-hot labels, matches softmax + categorical_crossentropy
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def _preprocess(image, label, augment):
    """Apply MobileNetV2 preprocessing (and augmentation if training)."""
    if augment:
        image = data_augmentation(image)
    # MobileNetV2 expects inputs scaled to [-1, 1]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def get_datasets():
    """
    Returns (train_ds, val_ds, test_ds, class_names) ready for model.fit().
    - train_ds: shuffled + augmented
    - val_ds / test_ds: not shuffled, not augmented
    """
    train_ds = _load_dataset_from_directory(TRAIN_DIR, shuffle=True)
    val_ds = _load_dataset_from_directory(VAL_DIR, shuffle=False)
    test_ds = _load_dataset_from_directory(TEST_DIR, shuffle=False)

    class_names = train_ds.class_names  # alphabetical order, matches label indices

    train_ds = train_ds.map(
        lambda x, y: _preprocess(x, y, augment=True), num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda x, y: _preprocess(x, y, augment=False), num_parallel_calls=AUTOTUNE
    )
    test_ds = test_ds.map(
        lambda x, y: _preprocess(x, y, augment=False), num_parallel_calls=AUTOTUNE
    )

    # Prefetch for performance - overlaps data loading with model execution
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    # Quick sanity check when run directly: prints shapes and class count
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print(f"Classes found: {len(class_names)}")
    for images, labels in train_ds.take(1):
        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")