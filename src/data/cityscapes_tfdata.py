# src/data/cityscapes_tfdata.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf

from .cityscapes_labels import LUT, G


def _decode_png(path: tf.Tensor, channels: int) -> tf.Tensor:
    b = tf.io.read_file(path)
    x = tf.image.decode_png(b, channels=channels)
    return x


def _load_example(img_path: tf.Tensor, mask_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    img:  (H,W,3) float32 [0,1]
    mask: (H,W)   int32 labelIds
    """
    img = _decode_png(img_path, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    mask = _decode_png(mask_path, channels=1)
    mask = tf.squeeze(mask, axis=-1)  # (H,W)
    mask = tf.cast(mask, tf.int32)
    return img, mask


def _resize_example(
    img: tf.Tensor,
    mask_label_ids: tf.Tensor,
    target_hw: Tuple[int, int],
) -> Tuple[tf.Tensor, tf.Tensor]:
    th, tw = target_hw
    img = tf.image.resize(img, [th, tw], method="bilinear")
    mask = tf.image.resize(mask_label_ids[..., None], [th, tw], method="nearest")
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)
    return img, mask

def _add_sample_weights_ignore_void(img: tf.Tensor, mask_group_ids: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    sample_weight: (H,W) float32
      - 0.0 pour void
      - 1.0 sinon
    """
    void_id = tf.cast(G["void"], tf.int32)
    w = tf.cast(tf.not_equal(mask_group_ids, void_id), tf.float32)  # (H,W)
    return img, mask_group_ids, w

def _map_to_groups(img: tf.Tensor, mask_label_ids: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    mask_label_ids -> mask_group_ids (0..7)
    Tout id hors LUT -> void
    """
    lut = tf.constant(LUT, dtype=tf.uint8)  # shape (max_id+1,)
    mask = tf.cast(mask_label_ids, tf.int32)

    max_id = tf.shape(lut)[0] - 1
    valid = tf.logical_and(mask >= 0, mask <= max_id)

    gathered = tf.gather(lut, tf.clip_by_value(mask, 0, max_id))
    gathered = tf.cast(gathered, tf.int32)

    void_id = tf.cast(G["void"], tf.int32)
    mapped = tf.where(valid, gathered, void_id)
    return img, mapped


def _augment_train(img: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    coin = tf.random.uniform([]) > 0.5
    img = tf.cond(coin, lambda: tf.image.flip_left_right(img), lambda: img)
    mask = tf.cond(coin, lambda: tf.image.flip_left_right(mask[..., None]), lambda: mask[..., None])
    mask = tf.squeeze(mask, axis=-1)

    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, lower=0.95, upper=1.05)
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img, mask


def make_cityscapes_ds(
    img_paths: list[str] | list[Path],
    mask_paths: list[str] | list[Path],
    *,
    target_hw: Tuple[int, int],
    batch_size: int,
    training: bool,
    use_augmentation: bool = True,
    shuffle_buffer: int = 1024,
    cache: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    if len(img_paths) != len(mask_paths):
        raise ValueError(f"img_paths and mask_paths must have same length: {len(img_paths)} vs {len(mask_paths)}")

    img_paths = [str(p) for p in img_paths]
    mask_paths = [str(p) for p in mask_paths]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    options = tf.data.Options()
    options.experimental_deterministic = not training
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda img, m: _resize_example(img, m, target_hw), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_map_to_groups, num_parallel_calls=tf.data.AUTOTUNE)

    if training and use_augmentation:
        ds = ds.map(_augment_train, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(_add_sample_weights_ignore_void, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds