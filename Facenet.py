import torch
import torch.nn as nn
from centerloss import CenterLoss
from itertools import chain

device = 'cuda'
class convolutional(nn.Module):
    def __init__(self,in_ch,out_ch,k_size,s,p,bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,k_size,s,p,bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1),)

    def forward(self, x):
        return self.layer(x)

class Ruidual(nn.Module):
    def __init__(self,in_ch):
        super().__init__()
        self.layer = nn.Sequential(
            convolutional(in_ch,in_ch//2,1,1,0),
            convolutional(in_ch//2, in_ch, 3, 1, 1),)
    def forward(self, x):
        return x + self.layer(x)

class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            convolutional( 3, 32, 3, 1, 0),
            convolutional(32, 64, 3, 2, 0),
            convolutional(64, 64, 3, 2, 0),
            convolutional(64, 64, 3, 2, 0),
            convolutional(64, 64, 3, 2, 0))
        self.linear = nn.Sequential(
            nn.Linear(64*14*14, 128),nn.PReLU(),
            nn.Linear(128, 2),nn.PReLU())
        self.linear1 = nn.Sequential(
            nn.Linear(2, 7),nn.LogSoftmax())
        self.centerloss = CenterLoss(2,7)
        self.loss = nn.NLLLoss()

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        self.cen = self.linear(x) #倒数第二层输出
        self.output = self.linear1(self.cen) #倒数第二层输出传入，倒数第一层输出
        return self.cen,self.output

    def getloss(self,y,al):
        centerloss = self.centerloss(self.cen,y)
        y=torch.tensor(list(chain(y))).long().to(device)
        outputloss = self.loss(self.output,y)
        loss = al*centerloss + (1-al)*outputloss
        return loss


if __name__ == '__main__':
    net = FaceNet()
    a = torch.arange(3*256*256).reshape(1,3,256,256).float()
    print(net(a))