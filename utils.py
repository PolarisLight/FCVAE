import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ADNet


def img2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)


def img2tensorfromfile(filename):
    assert filename is not None
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img2tensor(img)


class DataLoader(object):
    def __init__(self, traindatadir="LOLdataset\\our485\\",
                 testdatadir="LOLdataset\\eval15\\", batch_size=10):
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
        self.batch_size = batch_size

    def getTrainData(self, imnoise=0):
        low = []
        high = []
        for i in range(self.batch_size):
            list_shuffle = np.arange(self.TrCount)
            np.random.shuffle(list_shuffle)
            srcnum = list_shuffle[0]

            Lowimg = img2tensorfromfile(self.TrLowDir + self.TrLowName[srcnum])
            Highimg = img2tensorfromfile(self.TrHighDir + self.TrHighName[srcnum])

            low.append(Lowimg)
            high.append(Highimg)

        x = torch.cat(low, dim=0)
        if imnoise:
            noise = torch.abs(torch.randn(x.shape) / 25)
            x += noise
        y_true = torch.cat(high, dim=0)

        return x, y_true

    def getTestData(self):
        low = []
        high = []
        for i in range(self.batch_size):
            list_shuffle = np.arange(self.TeCount)
            np.random.shuffle(list_shuffle)
            srcnum = list_shuffle[0]

            Lowimg = img2tensorfromfile(self.TeLowDir + self.TeLowName[srcnum])
            Highimg = img2tensorfromfile(self.TeHighDir + self.TeHighName[srcnum])

            low.append(Lowimg)
            high.append(Highimg)

        x = torch.cat(low, dim=0)
        y_true = torch.cat(high, dim=0)

        return x, y_true


class deNoise(object):
    def __init__(self):
        torch.cuda.empty_cache()
        self.net = ADNet(channels=3, num_of_layers=17)
        device_ids = [0]
        self.model = torch.nn.DataParallel(self.net, device_ids=device_ids).cuda()
        self.model.load_state_dict(torch.load('model_70.pth'))
        self.model.eval()

    def run(self, Img):
        ISource = torch.Tensor(np.transpose(Img.copy(), (2, 0, 1))).unsqueeze(0)
        ISource = ISource
        ISource = ISource.cuda()
        with torch.no_grad():
            Out = torch.clamp(self.model(ISource), 0., 1.)
        Out = Out[0].cpu().numpy()
        Out = np.transpose(Out, (1, 2, 0))
        return Out


def guidedFilter(I, p, r, eps):
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


class fixcolor(object):
    def __init__(self):
        self.denoiser = deNoise()

    def run(self, S, Ill):
        S_double = S.numpy().transpose((1, 2, 0))
        R, G, B = S_double[:, :, 0], S_double[:, :, 1], S_double[:, :, 2]

        S = (S_double * 255).astype(np.uint8)
        L = np.max(S, 2)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, element)
        I = (L / 255.0).astype(np.float64)
        I = cv2.GaussianBlur(I, (5, 5), 0)
        R_r, R_g, R_b = R / I, G / I, B / I
        """
        R_max = R_r.max()
        G_max = R_g.max()
        B_max = R_b.max()
        R_r = R_r / R_max
        R_g = R_g / G_max
        R_b = R_b / B_max
        """
        I_in = np.zeros(S.shape)
        I_in[:, :, 0] = R_r
        I_in[:, :, 1] = R_g
        I_in[:, :, 2] = R_b

        I_out = I_in.copy()
        I_out = self.denoiser.run(I_in)
        # I_out = cv2.GaussianBlur(I_in, (5, 5), 0)
        # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # I_out = cv2.morphologyEx(I_out, cv2.MORPH_ERODE, element)
        # cv2.imshow("1", (I_out[:, :, -1] * 255).astype(np.uint8))
        # cv2.waitKey(30)
        R_r, R_g, R_b = I_out[:, :, 0], I_out[:, :, 1], I_out[:, :, 2]

        output = np.zeros(S.shape)
        Ill = Ill.squeeze(0).numpy().transpose((1, 2, 0))
        output[:, :, 0] = R_b  # * B_max
        output[:, :, 1] = R_g  # * G_max
        output[:, :, 2] = R_r  # * R_max
        output = output * Ill
        output = np.clip(output, 0, 1)
        return (output * 255).astype(np.uint8)


if __name__ == "__main__":
    cv2.boxFilter(1, 3)
