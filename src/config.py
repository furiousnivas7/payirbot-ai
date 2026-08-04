"""
PayirBot 2.0 - Configuration
Central place for paths, hyperparameters, and class info.
All other scripts import from this file.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

# Project root = the folder this file's parent (src/) lives in
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw dataset location (as downloaded + unzipped from Kaggle)
# Note: the zip nests the folder inside itself, hence the repeated name
RAW_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data", "raw",
    "New Plant Diseases Dataset(Augmented)",
    "New Plant Diseases Dataset(Augmented)"
)

RAW_TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")
RAW_VALID_DIR = os.path.join(RAW_DATA_DIR, "valid")   # this becomes our source for val + test split

# Processed dataset location (what prepare_data.py will create)
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

# Model output locations
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_model.h5")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "payirbot_model.tflite")

# Results output location
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ─────────────────────────────────────────────
# IMAGE / DATA SETTINGS
# ─────────────────────────────────────────────

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3

BATCH_SIZE = 32
SEED = 123

# The original dataset only ships train/ and valid/.
# We carve TEST_SPLIT_FROM_VALID of "valid/" into our own test set,
# the rest becomes our val set (see prepare_data.py).
TEST_SPLIT_FROM_VALID = 0.5   # 50% of "valid" -> val set, 50% -> held-out test set


# ─────────────────────────────────────────────
# CLASS NAMES (38 classes, matches folder names exactly)
# ─────────────────────────────────────────────

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)


# ─────────────────────────────────────────────
# TRAINING SETTINGS
# ─────────────────────────────────────────────

# Stage 1: train the new classification head only (base frozen)
STAGE1_EPOCHS = 10
STAGE1_LEARNING_RATE = 1e-3

# Stage 2: fine-tune the top layers of MobileNetV2
STAGE2_EPOCHS = 10
STAGE2_LEARNING_RATE = 1e-5
FINE_TUNE_AT_LAYER = 100   # unfreeze layers from this index onward

DROPOUT_RATE = 0.3