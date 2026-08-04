"""
PayirBot 2.0 - Data Preparation
Builds data/processed/{train,val,test} from the raw Kaggle dataset.

- train/  <- copied directly from raw train/ (already large & augmented)
- valid/  <- split 50/50 (by default) into our own val/ and test/
            so we get a genuinely held-out test set for honest evaluation

Run this once before training:
    python src/prepare_data.py
"""

import os
import random
import shutil

from config import (
    RAW_TRAIN_DIR,
    RAW_VALID_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    CLASS_NAMES,
    TEST_SPLIT_FROM_VALID,
    SEED,
)

random.seed(SEED)


def reset_dir(path):
    """Delete a directory if it exists, then recreate it empty."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def copy_train_set():
    """Copy the raw train/ folder into data/processed/train/, class by class."""
    print("Copying training set...")
    reset_dir(TRAIN_DIR)

    for class_name in CLASS_NAMES:
        src_class_dir = os.path.join(RAW_TRAIN_DIR, class_name)
        dst_class_dir = os.path.join(TRAIN_DIR, class_name)

        if not os.path.isdir(src_class_dir):
            print(f"  [WARNING] Missing class folder in raw train/: {class_name}")
            continue

        os.makedirs(dst_class_dir, exist_ok=True)
        images = os.listdir(src_class_dir)

        for img_name in images:
            src_path = os.path.join(src_class_dir, img_name)
            dst_path = os.path.join(dst_class_dir, img_name)
            shutil.copy2(src_path, dst_path)

        print(f"  {class_name}: {len(images)} images")

    print("Training set copy complete.\n")


def split_valid_into_val_and_test():
    """
    Split the raw valid/ folder into our own val/ and test/ sets.
    Split is done per-class so class balance is preserved in both sets.
    """
    print("Splitting validation set into val/ and test/...")
    reset_dir(VAL_DIR)
    reset_dir(TEST_DIR)

    for class_name in CLASS_NAMES:
        src_class_dir = os.path.join(RAW_VALID_DIR, class_name)

        if not os.path.isdir(src_class_dir):
            print(f"  [WARNING] Missing class folder in raw valid/: {class_name}")
            continue

        images = os.listdir(src_class_dir)
        random.shuffle(images)

        split_index = int(len(images) * (1 - TEST_SPLIT_FROM_VALID))
        val_images = images[:split_index]
        test_images = images[split_index:]

        val_class_dir = os.path.join(VAL_DIR, class_name)
        test_class_dir = os.path.join(TEST_DIR, class_name)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for img_name in val_images:
            shutil.copy2(
                os.path.join(src_class_dir, img_name),
                os.path.join(val_class_dir, img_name),
            )

        for img_name in test_images:
            shutil.copy2(
                os.path.join(src_class_dir, img_name),
                os.path.join(test_class_dir, img_name),
            )

        print(f"  {class_name}: {len(val_images)} val, {len(test_images)} test")

    print("Validation/test split complete.\n")


def print_summary():
    """Print a quick image-count summary for train/val/test."""
    def count_images(base_dir):
        total = 0
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(base_dir, class_name)
            if os.path.isdir(class_dir):
                total += len(os.listdir(class_dir))
        return total

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Train images: {count_images(TRAIN_DIR)}")
    print(f"Val images:   {count_images(VAL_DIR)}")
    print(f"Test images:  {count_images(TEST_DIR)}")
    print("=" * 50)


if __name__ == "__main__":
    print(f"Found {len(CLASS_NAMES)} classes.\n")
    copy_train_set()
    split_valid_into_val_and_test()
    print_summary()
    print("\nDone. Processed data is ready in data/processed/")