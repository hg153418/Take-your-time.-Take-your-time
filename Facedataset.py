import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import net
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from itertools import chain

Face_dir = r'E:\FaceRecognition\Face_Recognize_Sample\train\Zuojunchen1'
transform = transforms.Compose([
    transforms.ToTensor()
])

class Dataset(data.Dataset):

    def __init__(self,path):
        self.path = path
        self.dataset = []
        self.dataset.extend(os.listdir(path))
        #print(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = torch.Tensor(np.array([int(self.dataset[index].split('-')[0])]))
        img_path = os.path.join(self.path,self.dataset[index]) #图片的全部地址
        img_data = Image.open(img_path)
        data = torch.Tensor(np.transpose(img_data,(2,0,1))/255-0.5)

        return data,strs

if __name__ == '__main__':
    data = Dataset(Face_dir)
    print(data[1][0])


