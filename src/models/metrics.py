# src/models/metrics.py
from __future__ import annotations

import tensorflow as tf


class MeanIoUIgnoreVoid(tf.keras.metrics.Metric):
    """
    mIoU sur classes [0..num_classes-1] en ignorant void_id via sample_weight pixel-wise.
    - suppose y_true: (B,H,W) int
    - y_pred: (B,H,W,C) logits
    - sample_weight: (B,H,W) float 0/1 (0 pour void)
    """
    def __init__(self, num_classes: int, void_id: int, name="miou_no_void", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.void_id = int(void_id)
        self.cm = self.add_weight(
            name="confusion_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred logits -> classes
        y_pred_cls = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        y_true = tf.cast(y_true, tf.int32)
        y_pred_cls = tf.cast(y_pred_cls, tf.int32)

        # mask valide : ignore void + (optionnel) sample_weight
        valid = tf.not_equal(y_true, self.void_id)
        if sample_weight is not None:
            valid = tf.logical_and(valid, tf.cast(sample_weight > 0.0, tf.bool))

        y_true_f = tf.boolean_mask(y_true, valid)
        y_pred_f = tf.boolean_mask(y_pred_cls, valid)

        cm = tf.math.confusion_matrix(
            y_true_f,
            y_pred_f,
            num_classes=self.num_classes,
            dtype=tf.float32,
        )
        self.cm.assign_add(cm)

    def result(self):
        # IoU par classe = diag / (sum_row + sum_col - diag)
        diag = tf.linalg.diag_part(self.cm)
        sum_row = tf.reduce_sum(self.cm, axis=1)
        sum_col = tf.reduce_sum(self.cm, axis=0)
        denom = sum_row + sum_col - diag

        iou = tf.where(denom > 0, diag / denom, tf.zeros_like(diag))

        # On ignore void_id dans la moyenne
        mask = tf.ones([self.num_classes], dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(mask, [[self.void_id]], [False])
        iou_no_void = tf.boolean_mask(iou, mask)

        return tf.reduce_mean(iou_no_void)

    def reset_states(self):
        tf.keras.backend.set_value(self.cm, tf.zeros_like(self.cm))