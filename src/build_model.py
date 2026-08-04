"""
PayirBot 2.0 - Model Architecture
Defines the MobileNetV2 transfer-learning model.

Used by train.py - not meant to be run directly.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

from config import IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES, DROPOUT_RATE


def build_model():
    """
    Builds a MobileNetV2-based classifier for leaf disease detection.

    - Base: MobileNetV2 pretrained on ImageNet, top removed
    - Head: GlobalAveragePooling -> Dropout -> Dense(softmax)
    - Base is frozen by default; call set_fine_tune_layers() later
      to unfreeze top layers for Stage 2 fine-tuning.

    Returns:
        model: the full Keras Model (uncompiled)
        base_model: reference to the MobileNetV2 base (needed later for
                    freezing/unfreezing layers during fine-tuning)
    """
    input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # frozen for Stage 1

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name="payirbot_mobilenetv2")

    return model, base_model


def set_fine_tune_layers(base_model, fine_tune_at):
    """
    Unfreezes layers of the base model from `fine_tune_at` onward,
    keeping earlier layers frozen. Used for Stage 2 fine-tuning.

    Args:
        base_model: the MobileNetV2 base returned by build_model()
        fine_tune_at: layer index; layers before this stay frozen
    """
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    return base_model


if __name__ == "__main__":
    # Quick sanity check when run directly
    model, base_model = build_model()
    model.summary()
    print(f"\nTotal layers in base model: {len(base_model.layers)}")