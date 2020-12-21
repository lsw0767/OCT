"""
parallel parametric interpolation layer over mini-batch

in 2-d [n, n] array,
m, new_pos(m)-1 = K(m)
m, new_pos(m)   = 1-K(m)
where n=sig_len, 0<=m<=n, 0<=new_pos=int(k_linear_curve(m))<=sig_len-1, k=k_linear_curve(m)-int(k_linear_curve(m))

1. generate [batch, n, 2] array : [[K(0), 1-K(0)], ..., [K(n), 1-K(n)] * batch]
2. re-arange input signal to [batch, n, 2] : [[x[new_pos(0)-1], x[new_pos(0)]], ..., [x[new_pos(n)-1], x[new_pos(n)]] * batch]
3. element-wise multiplication
4. reduce_sum on last axis to generate interpolated signal [batch, n]
<<<<<<< HEAD:models/parametric_interpolation_v2.py

for differentiable indexing in (2), we used softmax-type kernel: k(new_pos(m))*x/sum(k(new_pos(m)))
where k(new_pos(m))[new_pos(m)] = 1, others are close to 0, so this differentiable formula can replace general indexing

=======

in our data, n=2048

"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class ParametricInterpolation(tf.keras.layers.Layer):
    def __init__(self, sig_len=2048, fully_trainable=True):
        super(ParametricInterpolation, self).__init__()
        self.sig_len = sig_len
        self.fully_trainable = fully_trainable

        self.scaler = tf.Variable(tf.constant([[1e11, 1e7, 1e3, 1e0, 1e0]]), trainable=False, dtype=np.float32)
        self.sig_idx = tf.constant(np.arange(0, self.sig_len, dtype=np.float32))
        self.matrix_size = tf.Variable(tf.constant([sig_len]), trainable=False)

    @tf.function
    def _generate_curve(self, params):
        return tf.vectorized_map(fn=lambda x: params[0]*x**4+params[1]*x**3+params[2]*x**2+params[3]*x**1+params[4], elems=self.sig_idx)

    @tf.function
    def call(self, x, params):
        params = tf.divide(params, self.scaler)
        curve_val = tf.vectorized_map(fn=lambda x: self._generate_curve(x), elems=params)
        curve_val_int = tf.round(curve_val)

        d = curve_val - curve_val_int
        d = tf.expand_dims(d, -1)
        d = tf.concat([1-d, d], axis=-1)

        if self.fully_trainable:
            new_pos = tfp.math.clip_by_value_preserve_gradient(self.sig_idx-curve_val_int, 1, 2047)
            new_pos = tf.expand_dims(new_pos, -1)
            x1_index_kernel = tf.exp(-tf.pow(new_pos-tf.cast(self.sig_idx, tf.float32), 4))
            x2_index_kernel = tf.exp(-tf.pow(new_pos-tf.cast(self.sig_idx, tf.float32)-1, 4))

            aranged_x1 = tf.tile(tf.expand_dims(x, axis=1), [1, 2048, 1])
            aranged_x1 = tf.reduce_sum(aranged_x1*x1_index_kernel/tf.reduce_sum(x1_index_kernel, axis=-1, keepdims=True), axis=-1)
            aranged_x2 = tf.tile(tf.expand_dims(x, axis=1), [1, 2048, 1])
            aranged_x2 = tf.reduce_sum(aranged_x2*x2_index_kernel/tf.reduce_sum(x2_index_kernel, axis=-1, keepdims=True), axis=-1)
        else:
            new_pos = tf.cast(tf.clip_by_value(self.sig_idx-curve_val_int, 1, 2047), tf.int32)
            aranged_x1 = tf.stack([tf.gather(tf.squeeze(batch_x), tf.squeeze(batch_pos))
                                   for batch_x, batch_pos in zip(tf.split(x, x.shape[0]), tf.split(new_pos, x.shape[0]))])
            aranged_x2 = tf.stack([tf.gather(tf.squeeze(batch_x), tf.squeeze(batch_pos-1))
                                   for batch_x, batch_pos in zip(tf.split(x, x.shape[0]), tf.split(new_pos, x.shape[0]))])

        x = tf.concat([tf.expand_dims(aranged_x1, -1), tf.expand_dims(aranged_x2, -1)], axis=-1)

        output = tf.multiply(x, d)
        output = tf.reduce_sum(output, axis=-1)

        return output, curve_val




