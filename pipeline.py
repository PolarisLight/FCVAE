import torch
import torchvision
import torch.optim
import os
import model
import numpy as np
from PIL import Image
import glob
import time
import cv2
import utils

point = "7"  # step
ckpt = 'save/Epoch' + point + '.pth'


class fixcolor(object):
    def __init__(self):
        self.denoiser = utils.deNoise()
        self.FCVAE = model.FCVAE()
        self.FCVAE.load_state_dict(torch.load(ckpt))

    def run(self, S):
        S_double = S.copy() / 255.0
        R, G, B = S_double[:, :, 0], S_double[:, :, 1], S_double[:, :, 2]

        L = np.max(S, 2)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, element)
        I = (L / 255.0).astype(np.float64)
        I = cv2.GaussianBlur(I, (5, 5), 0)
        R_r, R_g, R_b = R / I, G / I, B / I

        I_in = np.zeros(S.shape)
        I_in[:, :, 0] = R_r
        I_in[:, :, 1] = R_g
        I_in[:, :, 2] = R_b

        I_out = self.denoiser.run(I_in)
        S = torch.from_numpy(S_double).float()
        S = S.permute(2, 0, 1)
        S = S.unsqueeze(0)
        Ill, _, _ = self.FCVAE(S)
        R_r, R_g, R_b = I_out[:, :, 0], I_out[:, :, 1], I_out[:, :, 2]
        I_out = cv2.cvtColor(I_out, cv2.COLOR_BGR2RGB)
        output = np.zeros(I_in.shape)
        Ill = np.transpose(Ill.squeeze(0).numpy(), (1, 2, 0))
        output[:, :, 0] = R_b
        output[:, :, 1] = R_g
        output[:, :, 2] = R_r
        output = (output + Ill) / 2
        output = np.clip(output, 0, 1)
        return (output * 255).astype(np.uint8), (Ill * 255).astype(np.uint8), (I_out * 255).astype(np.uint8)


class lowlight(object):
    def __init__(self):
        self.fixer = fixcolor()

    def run(self, image_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        S = cv2.imread(image_path)
        S = cv2.cvtColor(S, cv2.COLOR_BGR2RGB)

        start = time.time()
        enhanced_image, illumination, reflaction = self.fixer.run(S)

        end_time = (time.time() - start)
        print(end_time)
        image_path = image_path.replace('low', 'final2')
        result_path = image_path
        if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
            os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
        cv2.imwrite(result_path, enhanced_image)
        cv2.imwrite(result_path.replace(".", 'i.'), cv2.cvtColor(illumination,cv2.COLOR_BGR2RGB))
        cv2.imwrite(result_path.replace(".", 'r.'), reflaction)


if __name__ == '__main__':
    pipeline = lowlight()
    with torch.no_grad():
        filePath = 'LoLdataset/eval15/low/'

        file_list = os.listdir(filePath)
        print(file_list)
        for file_name in file_list:
            test_list = glob.glob(filePath + file_name)
            for image in test_list:
                # image = image
                print(image)
                pipeline.run(image)
