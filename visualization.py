import torch
import torchvision
import torch.optim
import os
import model
import numpy as np
from PIL import Image
import glob
import cv2
import time

point = 5
ckpt = 'save/Epoch' + str(point) + '.pth'


def visualization(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)  # .cuda()

    FCVAE = model.FCVAE()  # .cuda()
    FCVAE.load_state_dict(torch.load(ckpt))
    start = time.time()
    enhanced_image, x1, x11 = FCVAE(data_lowlight)
    print(enhanced_image)
    x1list = torch.split(x11, 1, 1)
    print(x1list[0].shape, len(x1list))
    end_time = (time.time() - start)
    print(end_time)
    image_path = image_path.replace('low', 'visual-1')
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    for i in range(64):
        result_path = image_path.replace("1.", "1-section" + str(i) + ".")
        print("now writting %s" % result_path)
        midimg = x1list[i].squeeze(dim=0).squeeze(dim=0)
        midimg = midimg.numpy() * 255
        midimg = midimg.astype(np.uint8)
        cv2.imwrite(result_path, midimg)


if __name__ == '__main__':
    with torch.no_grad():
        filename = "LOLdataset/eval15/low/1.png"
        visualization(filename)
