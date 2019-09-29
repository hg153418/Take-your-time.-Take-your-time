import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
from Facenet import FaceNet
from Facedataset import Dataset
from itertools import chain
import os
import matplotlib.pyplot as plt
import numpy as np

Face_dir = r'E:\FaceRecognition\Face_Recognize_Sample\train\Zuojunchen1'
save_path = r'.\params\Facenet1.pt'
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
device = 'cuda'
color = ["red", "black", "yellow", "green", "pink", "gray", "lightgreen"]
# a = torchvision.datasets.ImageFolder(Face_dir,transform=transform)
# b = a.classes
# c = data.DataLoader(a,batch_size=52,shuffle=True,num_workers=2)

batch_size=32
Data = Dataset(Face_dir)
train_data = data.DataLoader(Data,batch_size=batch_size,shuffle=True)

FaceNet = FaceNet().to(device)
if os.path.exists(save_path):
    FaceNet.load_state_dict(torch.load(save_path))

Floss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(FaceNet.parameters())

accuracy = 0
FaceNet.train()

for epoch in range(1000):
    for i, (input, label) in enumerate(train_data):
        # print('label',label)
        # print('inout',input)
        x = input.to(device)
        # label = label.long()
        # y = y.long()
        cen, output = FaceNet(x.to(device))
        # label = torch.zeros(label.size()[0], 7).scatter_(1, label.view(-1, 1), 1)
        label = torch.Tensor(list(chain(label))).long().to(device)
        loss = FaceNet.getloss(label, 0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # print(cen)
        # print(label)
        # a = []
        # if epoch == 0:
        #     a.extend(cen)
        # print('a',a)

        if i % 20==0:
            print('[Tra - epoches - {}] loss:{}'.format(epoch,"%.7f" %loss.item()))

            with torch.no_grad():
                _, predicted = torch.max(output.data, 1)
                total = label.size(0)
                correct = (predicted == label).sum()
                accuracy = correct.float() / total
                print("[Tes - epoches - {0}]Accuracy:{1}%".format(epoch, (100 * accuracy)))
            torch.save(FaceNet.state_dict(), r"params/Facenet1.pt")
            print("**********************************")

        plt.ion()  # 开启绘画
        plt.clf()  # 清空内容
        for j in range(batch_size):
            mask = label == j
            #print(cen) #一轮输出得所有特征点包含x、y轴坐标
            _feature = cen[mask].cpu().data.numpy()
            plt.scatter(_feature[:,0], _feature[:,1], marker='o')
        plt.legend(['Tanhao', 'Wangxx', 'Wenqsh', 'Xiaowx','Yangzs', 'Zhangm', 'Zuojch'], loc='upper right')
        plt.title('CenterLoss—{0}'.format(i))
        plt.pause(0.1)

        if i == 100:
            plt.savefig('CenterLoss1.jpg')


























