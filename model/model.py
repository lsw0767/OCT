import tensorflow as tf

L2_NORM = 1e-5


class Regression(tf.keras.Model):
    def __init__(self, n_params, loss, is_cnn=True):
        super(Regression, self).__init__()
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

    @tf.function
    def __call__(self, x, is_training=True):
        x = tf.expand_dims(x, -1)
        for layer in self.l:
            x = layer(x)

        return x

    @tf.function
    def get_loss(self, x, y):
        return self.loss(y, self(x))

    @tf.function
    def train_on_batch(self, x, y):
        with tf.GradientTape() as t:
            loss = self.get_loss(x, y)
        grad = t.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return loss


class End2End(tf.keras.Model):
    def __init__(self, loss, is_cnn=True):
        super(End2End, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.loss = tf.keras.losses.MeanAbsoluteError() if loss=='mae' else tf.keras.losses.MeanSquaredError()

        if is_cnn:
            self.l = [
                tf.keras.layers.Conv1D(filters=4, kernel_size=9, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=8, kernel_size=9, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=16, kernel_size=9, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=32, kernel_size=9, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv1D(filters=1, kernel_size=9, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten()
            ]
        else:
            self.l = [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2048),
                tf.keras.layers.Dense(2048),
                tf.keras.layers.Dense(2048),
            ]

    @tf.function
    def __call__(self, x, is_training=True):
        x = tf.expand_dims(x, -1)
        for layer in self.l:
            x = layer(x)

        return x

    @tf.function
    def get_loss(self, x, y):
        return self.loss(y, self(x))

    @tf.function
    def train_on_batch(self, x, y):
        with tf.GradientTape() as t:
            loss = self.get_loss(x, y)
        grad = t.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return loss

