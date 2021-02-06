import torch.nn as nn
import torch


class FCVAE(nn.Module):

    def __init__(self):
        super(FCVAE, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Tanh()

        self.e_conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)

        self.convt6 = nn.ConvTranspose2d(256, 256, 3, 1, 1, bias=True)
        self.convt5 = nn.ConvTranspose2d(256, 128, 3, 1, 1, bias=True)
        self.convt4 = nn.ConvTranspose2d(128, 128, 3, 1, 1, bias=True)
        self.convt3 = nn.ConvTranspose2d(128, 64, 3, 1, 1, bias=True)
        self.convt2 = nn.ConvTranspose2d(64, 64, 3, 1, 1, bias=True)
        self.convt1 = nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.sigmoid(self.e_conv1(x))
        x2 = self.sigmoid(self.e_conv2(x1))

        x2 = self.maxpool(x2)

        x3 = self.sigmoid(self.e_conv3(x2))
        x4 = self.sigmoid(self.e_conv4(x3))

        x4 = self.maxpool(x4)

        x5 = self.sigmoid(self.e_conv5(x4))
        x6 = self.relu(self.e_conv6(x5))

        x7 = self.relu(self.convt6(x6))
        x8 = self.relu(self.convt5(x7))

        x8 = self.upsample(x8)

        x9 = self.relu(self.convt4(x8))
        x10 = self.relu(self.convt3(x9))

        x10 = self.upsample(x10)

        x11 = self.relu(self.convt2(x10))
        y = self.relu(self.convt1(x11))

        return y, x1, x11


class ADNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ADNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        return out2
