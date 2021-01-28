import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


class L_col(tf.keras.losses.Loss):
    def __init__(self):
        super(L_col, self).__init__()

    @tf.function
    def call(self, y_true, y_pred):
        mean_rgb = tf.reduce_mean(y_pred, [1, 2], keepdims=True)
        mr, mg, mb = tf.split(mean_rgb, num_or_size_splits=3, axis=-1)
        Drg = tf.pow(mr - mg, 2)
        Drb = tf.pow(mr - mb, 2)
        Dgb = tf.pow(mb - mg, 2)
        k = tf.pow(tf.pow(Drg, 2) + tf.pow(Drb, 2) + tf.pow(Dgb, 2), 0.5)
        return k


class L_spa(tf.keras.losses.Loss):
    def __init__(self):
        super(L_spa, self).__init__()
        """
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        """
        kernel_left = tf.constant(np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).astype(np.float32))
        kernel_left = tf.expand_dims(kernel_left, -1, name=None)
        self.weight_left = tf.expand_dims(kernel_left, -1, name=None)
        kernel_right = tf.constant(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).astype(np.float32))
        kernel_right = tf.expand_dims(kernel_right, -1, name=None)
        self.weight_right = tf.expand_dims(kernel_right, -1, name=None)
        kernel_up = tf.constant(np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).astype(np.float32))
        kernel_up = tf.expand_dims(kernel_up, -1, name=None)
        self.weight_up = tf.expand_dims(kernel_up, -1, name=None)
        kernel_down = tf.constant(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).astype(np.float32))
        kernel_down = tf.expand_dims(kernel_down, -1, name=None)
        self.weight_down = tf.expand_dims(kernel_down, -1, name=None)
        self.pool = tf.keras.layers.AveragePooling2D(4)

    @tf.function
    def call(self, enhance, org):
        b, c, h, w = org.shape
        org_mean = tf.reduce_mean(org, 3, keepdims=True)
        enhance_mean = tf.reduce_mean(enhance, 3, keepdims=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        D_org_letf = tf.nn.conv2d(org_pool, self.weight_left, strides=1, padding="VALID")
        D_org_right = tf.nn.conv2d(org_pool, self.weight_right, strides=1, padding="VALID")
        D_org_up = tf.nn.conv2d(org_pool, self.weight_up, strides=1, padding="VALID")
        D_org_down = tf.nn.conv2d(org_pool, self.weight_down, strides=1, padding="VALID")

        D_enhance_letf = tf.nn.conv2d(enhance_pool, self.weight_left, strides=1, padding="VALID")
        D_enhance_right = tf.nn.conv2d(enhance_pool, self.weight_right, strides=1, padding="VALID")
        D_enhance_up = tf.nn.conv2d(enhance_pool, self.weight_up, strides=1, padding="VALID")
        D_enhance_down = tf.nn.conv2d(enhance_pool, self.weight_down, strides=1, padding="VALID")

        D_left = tf.pow(D_org_letf - D_enhance_letf, 2)
        D_right = tf.pow(D_org_right - D_enhance_right, 2)
        D_up = tf.pow(D_org_up - D_enhance_up, 2)
        D_down = tf.pow(D_org_down - D_enhance_down, 2)
        # E = (D_left + D_right + D_up + D_down)
        E = 25 * (D_left + D_right + D_up + D_down)

        return E


class L_exp(tf.keras.losses.Loss):
    def __init__(self, patch_size=16, mean_val=0.6):
        super(L_exp, self).__init__()
        self.pool = layers.AveragePooling2D(patch_size)
        self.mean_val = mean_val

    @tf.function
    def call(self, y, x):
        # b, c, h, w = x.shape
        x = tf.reduce_mean(x, -1, keepdims=True)
        mean = self.pool(x)
        d = tf.reduce_mean(tf.pow(mean - tf.constant([self.mean_val]), 2))
        return d


class L_TV(tf.keras.losses.Loss):
    def __init__(self, TVLoss_weight=1.0):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    @tf.function
    def call(self, y, x):
        batch_size = tf.cast(tf.shape(x)[0], dtype=tf.float32)
        h_x = tf.shape(x)[1]
        w_x = tf.shape(x)[2]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        count_h = tf.cast(count_h, dtype=tf.float32)
        count_w = tf.cast(count_w, dtype=tf.float32)
        h_tv = tf.reduce_sum(tf.pow((x[:, 1:, :, :] - x[:, :h_x - 1, :, :]), 2))
        w_tv = tf.reduce_sum(tf.pow((x[:, :, 1:, :] - x[:, :, :w_x - 1, :]), 2))
        return self.TVLoss_weight * 2.0 * (h_tv / count_h + w_tv / count_w) / batch_size


class TotalLoss(tf.keras.losses.Loss):
    def __init__(self, name="totalloss", w_col=1.0, w_spa=1.0, w_exp=1.0, w_tv=1.0):
        super(TotalLoss, self).__init__(name=name)
        self.l_col = L_col()
        self.l_spa = L_spa()
        self.l_exp = L_exp()
        self.l_tv = L_TV()
        self.w_col = w_col
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_tv = w_tv

    @tf.function
    def call(self, y_true, y_pred):
        l_1 = self.w_col * self.l_col(y_true, y_pred)
        l_2 = self.w_spa * self.l_spa(y_true, y_pred)
        l_3 = self.w_exp * self.l_exp(y_true, y_pred)
        l_4 = self.w_tv * self.l_tv(y_true, y_pred)

        return l_1 + l_2 + l_3 + l_4
