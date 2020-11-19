import tensorflow as tf
import numpy as np

def k_linear_curve(idx, params):
    return params[0]*idx**4 + params[1]*idx**3 + params[2]*idx**2 + params[3]*idx +params[4]


class ParametricInterpolation(tf.keras.layers.Layer):
    def __init__(self, sig_len=2048):
        super(ParametricInterpolation, self).__init__()
        self.sig_len = sig_len
        self.scaler = tf.constant([1e13, 1e9, 1e5, 1e1, 1e2])

    def build(self, input_shape):
        self.sig_idx = tf.constant(np.arange(0, self.sig_len, dtype=np.float32))

    def call(self, x, params):
        params = tf.multiply(params, self.scaler)
        curve_val = tf.map_fn(fn=lambda t: k_linear_curve(t, params), elems=self.sig_idx, dtype=tf.float32)
        curve_val_int = tf.round(curve_val)
        k = curve_val_int - curve_val
        k = tf.expand_dims(k, -1)
        new_pos = tf.expand_dims(curve_val_int, -1)

        matrix_size = tf.constant([self.sig_len])

        elems = tf.concat([k, new_pos - 1], axis=1)
        intp_matrix_k_m_1 = tf.map_fn(fn=lambda x: tf.scatter_nd(tf.cast([[x[1]]], tf.int32), [x[0]], matrix_size),
                                      elems=elems)
        elems = tf.concat([1 - k, new_pos], axis=1)
        intp_matrix_k_m = tf.map_fn(fn=lambda x: tf.scatter_nd(tf.cast([[x[1]]], tf.int32), [x[0]], matrix_size),
                                    elems=elems)

        kernel = intp_matrix_k_m_1 + intp_matrix_k_m
        return tf.matmul(x, kernel)



