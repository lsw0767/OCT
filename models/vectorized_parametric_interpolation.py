"""
in 2-d nxn array,
m, new_pos(m)-1 = K_m
m, new_pos(m)   = (1-K_m)
where n=sig_len, 0<=m<=n, 0<=new_pos=int(k_linear_curve(m))<=sig_len-1, k=k_linear_curve(m)-int(k_linear_curve(m))

1. generate nx2 array :[[K_0, 1-K_0], ..., [K_n, 1-K_n]]
2. build nxn interpolation matrix by tf.scatter_nd

"""
import tensorflow as tf
import numpy as np


class ParametricInterpolation(tf.keras.layers.Layer):
    def __init__(self, sig_len=2048):
        super(ParametricInterpolation, self).__init__()
        self.sig_len = sig_len
        self.scaler = tf.Variable(tf.constant([[1e12, 1e8, 1e4, 1, 1e1]]), trainable=False, dtype=np.float32)
        self.sig_idx = tf.constant(np.arange(0, self.sig_len, dtype=np.float32))
        self.matrix_size = tf.Variable(tf.constant([sig_len]), trainable=False)

    # def build(self, input_shape):

    @tf.function
    def _generate_curve(self, params):
        return tf.vectorized_map(fn=lambda x: params[0]*x**4+params[1]*x**3+params[2]*x**2+params[3]*x**1+params[4], elems=self.sig_idx)

    @tf.function
    def call(self, x, params):
        params = tf.divide(params, self.scaler)
        curve_val = tf.vectorized_map(fn=lambda x: self._generate_curve(x), elems=params)
        curve_val_int = tf.round(curve_val)

        k = curve_val - curve_val_int
        k = tf.expand_dims(k, -1)
        k = tf.concat([1-k, k], axis=-1)

        new_pos = tf.cast(tf.clip_by_value(self.sig_idx-curve_val_int, 1, 2047), tf.int32)
        aranged_x1 = tf.stack([tf.gather(tf.squeeze(batch_x), tf.squeeze(batch_pos))
                               for batch_x, batch_pos in zip(tf.split(x, x.shape[0]), tf.split(new_pos, x.shape[0]))])
        aranged_x2 = tf.stack([tf.gather(tf.squeeze(batch_x), tf.squeeze(batch_pos-1))
                               for batch_x, batch_pos in zip(tf.split(x, x.shape[0]), tf.split(new_pos, x.shape[0]))])
        x = tf.concat([tf.expand_dims(aranged_x1, -1), tf.expand_dims(aranged_x2, -1)], axis=-1)

        output = tf.multiply(x, k)
        output = tf.reduce_sum(output, axis=-1)
        return output




