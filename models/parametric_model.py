import tensorflow as tf
import numpy as np

from vectorized_parametric_interpolation import ParametricInterpolation
from normalize_signal import NormalizeSignal

L2_NORM = 1e-5


class Model(tf.keras.Model):
    def __init__(self, n_params, loss, is_cnn=True, sig_len=2048):
        super(Model, self).__init__()
        self.sig_len = sig_len
        self.optimizer = tf.keras.optimizers.Adam(1e-2)
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
    def __call__(self, x, is_training=True, return_params=False):
        raw_sig = x
        x = tf.expand_dims(x, -1)
        for layer in self.l:
            x = layer(x)
        x = tf.tanh(x)
        params = x
        x = self.intp(raw_sig, x)
        x = self.norm(x)

        if return_params:
            return x, params
        else:
            return x

    @tf.function
    def get_loss(self, x, y, is_training=False):
        return self.loss(y, self(x, is_training=is_training))

    @tf.function
    def train_on_batch(self, x, y):
        with tf.GradientTape() as t:
            loss = self.get_loss(x, y, True)
        grad = t.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return loss

