import torch
import numpy as np
import torch.nn as nn
from Facenet import FaceNet
from PIL import Image
import os
import time

class test(nn.Module):
    def __init__(self,save_path):
        super(test, self).__init__()

        self.device = 'cuda'
        self.Facenet = FaceNet().to(self.device)
        self.Facenet.load_state_dict(torch.load(save_path))
        self.Facenet.eval()

    def forward(self, x):
        x = np.array(x)
        x = torch.Tensor(x).to(self.device).float()
        #print(x)
        x = x.permute(0,3,1,2)
        cen, output= self.Facenet(x)
        return cen, output

    def cos_sim(self,vector_a, vector_b):
        """
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a
        :param vector_b: 向量 b
        :return: sim
        """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim


if __name__ == '__main__':
    test_pic = r'.\Face_Recognize_Sample\train\Xiaoweixi\1.jpg'
    SJKU = r'E:\FaceRecognition\Face_Recognize_Sample\train\SJKU'
    save_path = r'.\params\Facenet1.pt'

    Start = time.time()
    Test = test(save_path)
    a = Image.open(test_pic)
    a = torch.unsqueeze(torch.Tensor(np.array(a)/255-0.5),dim=0)
    test_data = Test(a)

    List = []
    for i in range(7):
        with Image.open(os.path.join(SJKU,os.listdir(SJKU)[i])) as img:
            b = torch.unsqueeze(torch.Tensor(np.array(img) / 255 - 0.5), dim=0)
            SJKU_data = Test(b)
            #so
            M = test_data[0].cpu().detach().numpy()
            N = SJKU_data[0].cpu().detach().numpy()
            compare = Test.cos_sim(M,N) #测试向量与数据库向量做余弦相似度
        List.append(compare)

    TESTDATA = np.max(List,axis=0) #最大的cosine值（说明两个向量挨得最近）
    if TESTDATA <= 0.999:
        print("您与数据库资料不匹配，请您手动刷卡进入~")
    else:
        a = np.argmax(List,axis=0) #取出相应的类别
        Names = ['谭浩', '王潇潇', '温泉山', '肖维希','杨志森', '张密', '卓俊成']
        print("每类余弦相似度值：",List)
        print("余弦相似度最大值：",TESTDATA)
        print("识别该图片人脸为：",Names[a])

    End = time.time()
    Totaltime = End - Start
    print('检测时间：','%.2f'%Totaltime)


