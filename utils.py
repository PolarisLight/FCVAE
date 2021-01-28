import os

import cv2
import numpy as np
import tensorflow as tf


def img2tensor(img):
    img = img / 255.0
    img = tf.convert_to_tensor(img.astype(np.float32))
    img = tf.expand_dims(img, 0, name=None)
    return img


def img2tensorfromfile(filename):
    assert filename is not None
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img2tensor(img)


class DataLoader(object):
    def __init__(self, traindatadir="LOLdataset\\our485\\",
                 testdatadir="LOLdataset\\eval15\\"):
        self.TrLowDir = traindatadir + "low\\"
        self.TrHighDir = traindatadir + "high\\"
        self.TeLowDir = testdatadir + "low\\"
        self.TeHighDir = testdatadir + "high\\"
        self.TrLowName = os.listdir(self.TrLowDir)
        self.TrHighName = os.listdir(self.TrHighDir)
        self.TeLowName = os.listdir(self.TeLowDir)
        self.TeHighName = os.listdir(self.TeHighDir)
        self.TrCount = len(self.TrHighName)
        self.TeCount = len(self.TeHighName)

    def getTrainData(self, batch_size=10):
        low = []
        high = []
        for i in range(batch_size):
            list_shuffle = np.arange(self.TrCount)
            np.random.shuffle(list_shuffle)
            srcnum = list_shuffle[0]

            Lowimg = img2tensorfromfile(self.TrLowDir + self.TrLowName[srcnum])
            Highimg = img2tensorfromfile(self.TrHighDir + self.TrHighName[srcnum])

            low.append(Lowimg)
            high.append(Highimg)

        x = tf.concat(low, axis=0)
        y_true = tf.concat(high, axis=0)

        return x, y_true

    def getTestData(self, batch_size=10):
        low = []
        high = []
        for i in range(batch_size):
            list_shuffle = np.arange(self.TeCount)
            np.random.shuffle(list_shuffle)
            srcnum = list_shuffle[0]

            Lowimg = img2tensorfromfile(self.TeLowDir + self.TeLowName[srcnum])
            Highimg = img2tensorfromfile(self.TeHighDir + self.TeHighName[srcnum])

            low.append(Lowimg)
            high.append(Highimg)

        x = tf.concat(low, axis=0)
        y_true = tf.concat(high, axis=0)

        return x, y_true


if __name__ == "__main__":
    pass