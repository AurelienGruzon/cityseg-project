from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as L


def conv_block(x, filters: int):
    x = L.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = L.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_vgg16_unet(
    input_shape=(256, 512, 3),
    num_classes: int = 8,
    freeze_encoder: bool = True,
) -> tf.keras.Model:
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    # Skip connections
    s1 = base_model.get_layer("block1_conv2").output   # 256x512
    s2 = base_model.get_layer("block2_conv2").output   # 128x256
    s3 = base_model.get_layer("block3_conv3").output   # 64x128
    s4 = base_model.get_layer("block4_conv3").output   # 32x64

    b1 = base_model.get_layer("block5_conv3").output   # 16x32

    if freeze_encoder:
        for layer in base_model.layers:
            layer.trainable = False

    # Decoder
    d1 = L.UpSampling2D((2, 2))(b1)         # 32x64
    d1 = L.Concatenate()([d1, s4])
    d1 = conv_block(d1, 512)

    d2 = L.UpSampling2D((2, 2))(d1)         # 64x128
    d2 = L.Concatenate()([d2, s3])
    d2 = conv_block(d2, 256)

    d3 = L.UpSampling2D((2, 2))(d2)         # 128x256
    d3 = L.Concatenate()([d3, s2])
    d3 = conv_block(d3, 128)

    d4 = L.UpSampling2D((2, 2))(d3)         # 256x512
    d4 = L.Concatenate()([d4, s1])
    d4 = conv_block(d4, 64)

    outputs = L.Conv2D(num_classes, 1, padding="same", activation=None)(d4)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs, name="vgg16_unet")