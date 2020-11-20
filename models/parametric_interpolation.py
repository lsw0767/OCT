"""
in 2-d nxn array,
m, new_pos(m)-1 = K_m
m, new_pos(m)   = (1-K_m)
where n=sig_len, 0<=m<=n, 0<=new_pos=int(k_linear_curve(m))<=sig_len-1, k=k_linear_curve(m)-int(k_linear_curve(m))

1. generate nx2 array :[[K_0, 1-K_0], ..., [K_n, 1-K_n]]
2. build nxn interpolation matrix by tf.scatter_nd

todo: vectorization
"""
import tensorflow as tf
import numpy as np

def k_linear_curve(idx, params):
    return params[0]*idx**4 + params[1]*idx**3 + params[2]*idx**2 + params[3]*idx +params[4]


class ParametricInterpolation(tf.keras.layers.Layer):
    def __init__(self, sig_len=2048):
        super(ParametricInterpolation, self).__init__()
        self.sig_len = sig_len
        self.scaler = tf.Variable(tf.constant([1e12, 1e8, 1e4, 1, 1e1]), trainable=False, dtype=np.float32)
        self.sig_idx = tf.Variable(tf.constant(np.arange(0, self.sig_len, dtype=np.float32)), trainable=False)
        self.matrix_size = tf.Variable(tf.constant([sig_len]), trainable=False)

    # def build(self, input_shape):

    @tf.function
    def _batchwise_intp(self, x, params):
        x = tf.squeeze(x, 0)
        params = tf.squeeze(params, 0)
        params = tf.divide(params, self.scaler)
        curve_val = tf.map_fn(fn=lambda t: k_linear_curve(t, params), elems=self.sig_idx, dtype=tf.float32)
        curve_val_int = tf.round(curve_val)
        k = curve_val - curve_val_int
        new_pos = tf.cast(tf.clip_by_value(self.sig_idx-curve_val_int, 1, 2047), tf.int32)

        k = tf.expand_dims(k, -1)
        new_pos = tf.expand_dims(new_pos, -1)

        aranged_x1 = tf.expand_dims(tf.gather_nd(x, new_pos), -1)
        aranged_x2 = tf.expand_dims(tf.gather_nd(x, new_pos-1), -1)
        x = tf.concat([aranged_x1, aranged_x2], axis=1)
        k = tf.concat([1-k, k], axis=1)

        output = tf.multiply(x, k)
        output = tf.reduce_sum(output, axis=1)
        return output

    @tf.function
    def call(self, x, params):
        batch_size = params.shape[0]
        split_x = tf.split(x, batch_size)
        split_params = tf.split(params, batch_size)
        output = [self._batchwise_intp(single_x, single_params) for single_x, single_params in zip(split_x, split_params)]
        # for single_x, single_params in zip(split_x, split_params):
        #     output.append(self._batchwise_intp(single_x, single_params))
        output = tf.stack(output)
        return output




