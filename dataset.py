import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
import torch
import torch.nn as nn
from utils import data_augmentation


def normalize(data):
    return data / 255.


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    # print endc
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    high_files = glob.glob(os.path.join(data_path, 'high', '*'))
    low_files = glob.glob(os.path.join(data_path, 'low', '*'))
    high_files.sort()
    low_files.sort()
    h5high = h5py.File('high.h5', 'w')
    h5low = h5py.File('low.h5', 'w')
    train_num = 0
    for i in range(len(high_files)):
        img_high = cv2.imread(high_files[i])
        img_low = cv2.imread(low_files[i])
        h, w, c = img_high.shape
        for k in range(len(scales)):
            img_high = cv2.resize(img_high, (int(h * scales[k]), int(w * scales[k])), interpolation=cv2.INTER_CUBIC)
            img_high = torch.tensor(img_high)
            img_high = img_high.permute(2, 0, 1)
            img_high = img_high.numpy()
            img_high = np.float32(normalize(img_high))
            patches_high = Im2Patch(img_high, win=patch_size, stride=stride)

            img_low = cv2.resize(img_low, (int(h * scales[k]), int(w * scales[k])), interpolation=cv2.INTER_CUBIC)
            img_low = torch.tensor(img_low)
            img_low = img_low.permute(2, 0, 1)
            img_low = img_low.numpy()
            img_low = np.float32(normalize(img_low))
            patches_low = Im2Patch(img_low, win=patch_size, stride=stride)

            print("file: %s,%s scale %.1f # samples: %d" % (
            high_files[i], low_files[i], scales[k], patches_high.shape[3] * aug_times))

            for n in range(patches_high.shape[3]):
                rand = np.random.randint(1, 8)
                data_high = patches_high[:, :, :, n].copy()
                data_low = patches_low[:, :, :, n].copy()
                data_high = data_augmentation(data_high, rand)
                data_low = data_augmentation(data_low, rand)
                h5high.create_dataset(str(train_num), data=data_high)
                h5low.create_dataset(str(train_num), data=data_low)
                train_num += 1
    h5high.close()
    h5low.close()
    print('training set, # samples %d\n' % train_num)


"""
    # val
    print('\nprocess validation data')
    files = glob.glob(os.path.join(data_path, 'CBSD68', '*.bmp'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)
        img = img.numpy()
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    
    print('val set, # samples %d\n' % val_num)
"""


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5high = h5py.File('high.h5', 'r')
            h5low = h5py.File('low.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        highdata = np.array(h5high[key])
        lowdata = np.array(h5low[key])
        h5high.close()
        h5low.close()
        return torch.Tensor(lowdata), torch.Tensor(highdata)
