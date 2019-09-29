import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

device = 'cuda'
class CenterLoss(nn.Module):

    def __init__(self, feature_num, cls_num):
        super(CenterLoss,self).__init__()

        self.cls_num = cls_num
        """创建中心用nn.Parameter(),保证反算传播时可以算"""
        self.center = nn.Parameter(torch.randn(cls_num, feature_num)) #cls_num分类数，feature_num特征点

    def forward(self, xs, ys):
        xs = F.normalize(xs) #归一化
        ys = list(chain(ys)) #把二维降到一维，下面充当索引需要1维
        ys = torch.Tensor(ys)

        center_exp = self.center.index_select(dim=0, index=ys.long().to(device))
        # print(xs.size())
        # print(center_exp.size())
        count = torch.histc(ys.float(), bins=self.cls_num, min=0, max=self.cls_num-1)
        count_dis = count.index_select(dim=0, index=ys.long())
        L = torch.sum(torch.sqrt(torch.sum((xs.to(device) - center_exp.to(device)) ** 2, dim=1)) / count_dis.float().to(device))

        return L



