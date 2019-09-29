import torch
import torch.nn as nn
import math


class FaceNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layerO = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 2, 1),
            nn.PReLU(),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.PReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.PReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(128*2*2, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU()
        )
        self.con = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.offset = nn.Sequential(
            nn.Linear(256, 4)
        )
        self.landmark = nn.Linear(256, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.layerO(x)
        linear = self.linear(x.view(x.size(0), -1))
        con = self.con(linear)
        offset = self.offset(linear)
        landmark = self.landmark(linear)
        return con, offset, landmark







