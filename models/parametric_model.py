import tensorflow as tf
import numpy as np

from parallel_parametric_interpolation import ParametricInterpolation
from normalize_signal import NormalizeSignal

L2_NORM = 1e-5


class Model(tf.keras.Model):
    def __init__(self, n_params, loss, is_cnn=True, sig_len=2048):
        super(Model, self).__init__()
        self.sig_len = sig_len
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.loss = tf.keras.losses.MeanAbsoluteError() if loss=='mae' else tf.keras.losses.MeanSquaredError()

        if is_cnn:
            self.l = [
                tf.keras.layers.Conv1D(filters=4, kernel_size=9, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=8, kernel_size=9, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=16, kernel_size=9, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=32, kernel_size=9, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=64, kernel_size=9, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(n_params)
            ]
        else:
            self.l = [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(32),
                tf.keras.layers.Dense(n_params)
            ]

        self.intp = ParametricInterpolation(sig_len)
        self.norm = NormalizeSignal()

    @tf.function
    def __call__(self, x, is_training=True, return_curve=False):
        raw_sig = x
        x = tf.expand_dims(x, -1)
        for layer in self.l:
            x = layer(x)
        x = tf.tanh(x)
        params = x
        x, curve = self.intp(raw_sig, x)
        x = self.norm(x)

        if return_curve:
            return x, params, curve
        else:
            return x

    @tf.function
    def get_loss(self, x, y):
        return self.loss(y, x)

    @tf.function
    def get_curve_loss(self, curve, curve_target):
        return self.get_loss(curve, curve_target)

    @tf.function
    def train_on_batch(self, x, y):
        with tf.GradientTape() as t:
            # loss = self.get_loss(x, y, True)
            y_, _, curve = self(x, is_training=True, return_curve=True)
            loss = self.get_loss(y_, y)
            # curve_loss = self.get_curve_loss(curve, curve_target)
            """
            with t.stop_recording():
                Todo: calculate peak index for index gradient 
            """
            # total_loss = loss + curve_loss
        grad = t.gradient(loss, self.trainable_variables)
        # for v in self.trainable_variables:
        #     dcurve_dx = tf.gradients(curve, v)
        #     print(dcurve_dx)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return loss

