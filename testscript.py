import utils
import cv2
import torch
import numpy as np

img = utils.img2tensorfromfile("LoLdataset/eval15/low/23.png")
img = img.squeeze(0)

noise = torch.abs(torch.randn(img.shape)/25)
img += noise

img = img.numpy() * 255
img = img.transpose((1, 2, 0))
img = img.astype(np.uint8)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow("1", img)
cv2.waitKey(0)
