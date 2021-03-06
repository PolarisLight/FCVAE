import os

import torch
import torch.optim
import winsound
import Myloss
import matplotlib.pyplot as plt
import model
import utils
import tqdm

times = 2
_ANHOUR = 4
batch_size = 2
lr = 1e-3
num_epochs = 1 if int(times * _ANHOUR) == 0 else int(times * _ANHOUR)

step = 1000

weight_tv = 20
weight_col = 20
weight_spa = 100
weight_exp = 10

all_loss = []


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train():
    global lr
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    FCVAE = model.FCVAE().cuda()
    # FCVAE.load_state_dict(torch.load('save/Epoch5.pth'))
    # FCVAE.apply(weights_init)

    loader = utils.DataLoader(batch_size=batch_size)

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()

    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(FCVAE.parameters(), lr=lr)

    FCVAE.train()

    for epoch in range(num_epochs):
        if epoch % 6 == 0:
            lr = lr / 2
            optimizer = torch.optim.Adam(FCVAE.parameters(), lr=lr)
        with tqdm.tqdm(total=step) as pbar:
            for i in range(step):
                img_lowlight, img_highlight = loader.getTrainData()
                img_lowlight = img_lowlight.cuda()
                img_highlight = img_highlight.cuda()
                enhanced_image, _, _ = FCVAE(img_lowlight)
                Loss_TV = weight_tv * L_TV(enhanced_image)

                loss_spa = weight_spa * torch.mean(L_spa(enhanced_image, img_lowlight))

                loss_col = weight_col * torch.mean(L_color(enhanced_image, img_highlight))

                loss_exp = weight_exp * torch.mean(L_exp(enhanced_image))

                loss_reg = torch.sum(torch.pow(img_highlight - enhanced_image, 2)) / 100

                loss = Loss_TV + loss_spa + loss_exp + loss_reg  # + loss_col
                all_loss.append(loss)
                del Loss_TV, loss_spa, loss_col, loss_exp, loss_reg, img_lowlight, img_highlight, enhanced_image
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(FCVAE.parameters(), 0.1)
                optimizer.step()
                pbar.set_description("epoch = %d,"
                                     "loss = %.6f" % (epoch + 1, loss.item()))
                pbar.update(1)

        torch.save(FCVAE.state_dict(), "save/" + "Epoch" + str(epoch) + '.pth')
    axis = [x for x in range(num_epochs * step)]
    plt.plot(axis, all_loss)
    plt.show()
    duration = 1000  # millisecond
    freq = 880  # Hz
    winsound.Beep(freq, duration)


if __name__ == "__main__":
    train()
