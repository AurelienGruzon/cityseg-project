# src/models/unet.py
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as L


def _conv_block(x, filters: int, dropout: float = 0.0):
    x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    if dropout and dropout > 0:
        x = L.Dropout(dropout)(x)
    return x


def build_unet(
    input_shape=(256, 512, 3),
    num_classes: int = 8,
    base_filters: int = 32,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """
    U-Net baseline. Output = logits (no softmax), shape (H,W,num_classes)
    """
    inputs = L.Input(shape=input_shape)

    # Encoder
    c1 = _conv_block(inputs, base_filters, dropout=0.0)
    p1 = L.MaxPooling2D()(c1)

    c2 = _conv_block(p1, base_filters * 2, dropout=0.0)
    p2 = L.MaxPooling2D()(c2)

    c3 = _conv_block(p2, base_filters * 4, dropout=dropout)
    p3 = L.MaxPooling2D()(c3)

    c4 = _conv_block(p3, base_filters * 8, dropout=dropout)
    p4 = L.MaxPooling2D()(c4)

    # Bottleneck
    bn = _conv_block(p4, base_filters * 16, dropout=dropout)

    # Decoder
    u4 = L.UpSampling2D()(bn)
    u4 = L.Concatenate()([u4, c4])
    c5 = _conv_block(u4, base_filters * 8, dropout=dropout)

    u3 = L.UpSampling2D()(c5)
    u3 = L.Concatenate()([u3, c3])
    c6 = _conv_block(u3, base_filters * 4, dropout=dropout)

    u2 = L.UpSampling2D()(c6)
    u2 = L.Concatenate()([u2, c2])
    c7 = _conv_block(u2, base_filters * 2, dropout=0.0)

    u1 = L.UpSampling2D()(c7)
    u1 = L.Concatenate()([u1, c1])
    c8 = _conv_block(u1, base_filters, dropout=0.0)

    # Logits
    logits = L.Conv2D(num_classes, 1, padding="same", name="logits")(c8)

    return tf.keras.Model(inputs, logits, name="unet_baseline")