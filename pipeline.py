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

point = 5
ckpt = 'save/Epoch' + str(point) + '.pth'


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)  # .cuda()

    FCVAE = model.FCVAE()  # .cuda()
    fixer = utils.fixcolor()
    FCVAE.load_state_dict(torch.load(ckpt))
    start = time.time()
    enhanced_image, _, _ = FCVAE(data_lowlight)
    enhanced_image = fixer.run(data_lowlight.squeeze(0), enhanced_image)
    end_time = (time.time() - start)
    # print(end_time)
    image_path = image_path.replace('low', 'final')
    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    cv2.imwrite(result_path, enhanced_image)
    # torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
    with torch.no_grad():
        filePath = 'LoLdataset/eval15/low/'

        file_list = os.listdir(filePath)
        print(file_list)
        for file_name in file_list:
            test_list = glob.glob(filePath + file_name)
            print(test_list)
            for image in test_list:
                # image = image
                print(image)
                lowlight(image)
