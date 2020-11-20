import tensorflow as tf


class NormalizeSignal(tf.keras.layers.Layer):
    def __init__(self):
        super(NormalizeSignal, self).__init__()

    @tf.function
    def call(self, x):
        min_val = tf.reduce_min(x, axis=1, keepdims=True)
        x = tf.subtract(x, min_val)
        max_val = tf.reduce_max(x, axis=1, keepdims=True)
        x = tf.divide(x, max_val)
        return x




