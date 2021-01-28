import tensorflow.keras as keras

from codes.MyLoss import *


class FCVAE(keras.Model):

    def __init__(
            self,
            activation="relu",
            name="FCVAE",

            **kwargs
    ):
        super(FCVAE, self).__init__(name=name, **kwargs)
        self.activation = activation
        self.Loss = TotalLoss()
        self.conv1 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   strides=1,
                                   activation=activation,
                                   padding="same")
        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   strides=1,
                                   activation=activation,
                                   padding="same")

        self.pooling1 = layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv3 = layers.Conv2D(filters=128,
                                   kernel_size=[3, 3],
                                   strides=1,
                                   activation=activation,
                                   padding="same")
        self.conv4 = layers.Conv2D(filters=128,
                                   kernel_size=[3, 3],
                                   strides=1,
                                   activation=activation,
                                   padding="same")

        self.pooling2 = layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv5 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   strides=1,
                                   activation=activation,
                                   padding="same")
        self.convt5 = layers.Conv2DTranspose(filters=128,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             activation=activation,
                                             padding="same")

        self.upsampling2 = layers.UpSampling2D(size=(2, 2))

        self.convt4 = layers.Conv2DTranspose(filters=128,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             activation=activation,
                                             padding="same")
        self.convt3 = layers.Conv2DTranspose(filters=64,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             activation=activation,
                                             padding="same")
        self.upsampling1 = layers.UpSampling2D(size=(2, 2))

        self.convt2 = layers.Conv2DTranspose(filters=64,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             activation=activation,
                                             padding="same")
        self.convt1 = layers.Conv2DTranspose(filters=3,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             activation=activation,
                                             padding="same")

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pooling1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling2(x)
        x = self.conv5(x)
        x = self.convt5(x)
        x = self.upsampling2(x)
        x = self.convt4(x)
        x = self.convt3(x)
        x = self.upsampling1(x)
        x = self.convt2(x)
        x = self.convt1(x)
        return x

    @tf.function
    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

        y_pred = tf.cast(y_pred, tf.float32)

        # l_col
        mean_rgb = tf.reduce_mean(y_pred, [1, 2], keepdims=True)
        mr, mg, mb = tf.split(mean_rgb, num_or_size_splits=3, axis=-1)
        Drg = tf.pow(mr - mg, 2)
        Drb = tf.pow(mr - mb, 2)
        Dgb = tf.pow(mb - mg, 2)
        l_col = tf.pow(tf.pow(Drg, 2) + tf.pow(Drb, 2) + tf.pow(Dgb, 2), 0.5)

        # l_spa
        weight_left = tf.expand_dims(
            tf.expand_dims(tf.constant(np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).astype(np.float32)), -1, name=None)
            , -1, name=None)
        weight_right = tf.expand_dims(
            tf.expand_dims(tf.constant(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).astype(np.float32)), -1, name=None)
            , -1, name=None)
        weight_up = tf.expand_dims(
            tf.expand_dims(tf.constant(np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).astype(np.float32)), -1, name=None)
            , -1, name=None)
        weight_down = tf.expand_dims(
            tf.expand_dims(tf.constant(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).astype(np.float32)), -1, name=None)
            , -1, name=None)

        org_mean = tf.reduce_mean(x, 3, keepdims=True)
        enhance_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        org_pool = tf.nn.avg_pool2d(org_mean, 4, 4, "SAME")
        enhance_pool = tf.nn.avg_pool2d(enhance_mean, 4, 4, "SAME")
        D_org_letf = tf.nn.conv2d(org_pool, weight_left, strides=1, padding="VALID")
        D_org_right = tf.nn.conv2d(org_pool, weight_right, strides=1, padding="VALID")
        D_org_up = tf.nn.conv2d(org_pool, weight_up, strides=1, padding="VALID")
        D_org_down = tf.nn.conv2d(org_pool, weight_down, strides=1, padding="VALID")

        D_enhance_letf = tf.nn.conv2d(enhance_pool, weight_left, strides=1, padding="VALID")
        D_enhance_right = tf.nn.conv2d(enhance_pool, weight_right, strides=1, padding="VALID")
        D_enhance_up = tf.nn.conv2d(enhance_pool, weight_up, strides=1, padding="VALID")
        D_enhance_down = tf.nn.conv2d(enhance_pool, weight_down, strides=1, padding="VALID")

        D_left = tf.pow(D_org_letf - D_enhance_letf, 2)
        D_right = tf.pow(D_org_right - D_enhance_right, 2)
        D_up = tf.pow(D_org_up - D_enhance_up, 2)
        D_down = tf.pow(D_org_down - D_enhance_down, 2)

        l_spa = 25 * (D_left + D_right + D_up + D_down)

        # l_exp
        patch_size = 16
        mean_val = 0.6
        pool = layers.AveragePooling2D(patch_size)
        y_pred = tf.reduce_mean(y_pred, -1, keepdims=True)
        mean = pool(y_pred)
        l_exp = tf.reduce_mean(tf.pow(mean - tf.constant([mean_val]), 2))

        # l_tv
        TVLoss_weight = 1.0
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
        h_x = tf.shape(y_pred)[1]
        w_x = tf.shape(y_pred)[2]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        count_h = tf.cast(count_h, dtype=tf.float32)
        count_w = tf.cast(count_w, dtype=tf.float32)
        h_tv = tf.reduce_sum(tf.pow((x[:, 1:, :, :] - x[:, :h_x - 1, :, :]), 2))
        w_tv = tf.reduce_sum(tf.pow((x[:, :, 1:, :] - x[:, :, :w_x - 1, :]), 2))
        l_tv = TVLoss_weight * 2.0 * (h_tv / count_h + w_tv / count_w) / batch_size

        # total loss
        loss = 1.0 * l_col + 1.0 * l_spa + 1.0 * l_exp + 1.0 * l_tv

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}
