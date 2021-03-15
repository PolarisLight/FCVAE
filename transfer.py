import os
import winsound

import matplotlib.pyplot as plt
import torch
import torch.optim
import tqdm
import model

import utils
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import cv2

times = 3
_ANHOUR = 1
batch_size = 6
lr = 1e-3
num_epochs = 1 if int(times * _ANHOUR) == 0 else int(times * _ANHOUR)

step = 100

all_loss = []
all_loss1 = []
all_loss2 = []
final_loss = []


def nn_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = torch.nn.Conv2d(3, 3, 3, bias=False, padding=1)
    # 定义sobel算子参数
    sobel_kernel = np.array([[1 / 8, 2 / 8, 1 / 8], [0, 0, 0], [-1 / 8, -2 / 8, -1 / 8]] * 3,
                            dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 3, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # 对图像进行卷积操作
    edge_detect = conv_op(torch.autograd.Variable(im))
    return edge_detect


def train():
    global lr, num_epochs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    torch.cuda.empty_cache()

    net = model.ADNet(channels=3, num_of_layers=17)
    criterion = torch.nn.MSELoss(size_average=False)

    device_ids = [0, 1]
    model1 = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()

    optimizer = torch.optim.Adam(model1.parameters(), lr=lr)

    ckpt = torch.load('model_70.pth')
    model1.load_state_dict(ckpt)

    loader = utils.transDataLoader(batch_size=batch_size)

    for epoch in range(num_epochs):
        if epoch > 0:
            lr = lr / 2
            optimizer = torch.optim.Adam(model1.parameters(), lr=lr)
        with tqdm.tqdm(total=step) as pbar:
            for i in range(step):
                img_lowlight, img_highlight = loader.getTrainData(imnoise=0)
                img_lowlight = img_lowlight.cuda()
                img_highlight = img_highlight.cuda()

                model1.train()
                enhanced_image = torch.clamp(model1(img_lowlight), 0., 1.)
                show_img = enhanced_image[0].detach().cpu().numpy()

                enhance_edge = nn_conv2d(enhanced_image.cpu())

                high_edge = nn_conv2d(img_highlight.cpu())

                show_img = np.transpose(show_img, (1, 2, 0))
                cv2.imshow("output", (show_img * 255).astype(np.uint8))

                # enhance_edge = torch.clamp(enhance_edge, 0., 1.)
                # high_edge_sum = torch.sum(high_edge, (2, 3), keepdim=True)

                loss1 = criterion(enhance_edge, high_edge) / (high_edge.shape[1] * high_edge.shape[2] * 2)

                enhance_mean = torch.mean(enhanced_image, (2, 3))
                low_mean = torch.mean(img_lowlight, (2, 3))
                loss2 = criterion(enhance_mean, low_mean)

                loss = loss1 + loss2
                all_loss.append(loss)
                all_loss1.append(loss1)
                all_loss2.append(loss2)
                optimizer.zero_grad()
                loss.backward()

                i1 = enhance_edge[0].cpu().detach().numpy()
                i2 = high_edge[0].cpu().detach().numpy()

                i1 = np.transpose(i1, (1, 2, 0))
                cv2.imshow("enhanced", (i1 * 255).astype(np.uint8))

                i2 = np.transpose(i2, (1, 2, 0))
                cv2.imshow("target", (i2 * 255).astype(np.uint8))
                cv2.waitKey(10)

                torch.nn.utils.clip_grad_norm_(model1.parameters(), 0.1)
                optimizer.step()

                pbar.set_description("epoch = %d,"
                                     "loss1 = %.6f,"
                                     "loss2 = %.6f " % (epoch + 1, loss1.item(), loss2.item()))
                pbar.update(1)
        torch.save(model1.state_dict(), "model_" + str(epoch) + '.pth')
        final_loss.append(all_loss[-1].item())
        print((final_loss[epoch] - final_loss[epoch - 1]) ** 2)
        """
        if epoch > 0:
            if final_loss[epoch] > final_loss[epoch - 1] or (final_loss[epoch] - final_loss[epoch - 1]) ** 2 < 1000:
                num_epochs = epoch + 1
                print("early stop")
                break"""
    model1.eval()
    axis = [x for x in range(num_epochs * step)]
    # plt.plot(axis, all_loss)
    plt.plot(axis, all_loss1)
    plt.plot(axis, all_loss2)
    plt.show()
    duration = 1000  # millisecond
    freq = 880  # Hz
    winsound.Beep(freq, duration)


if __name__ == "__main__":
    train()
