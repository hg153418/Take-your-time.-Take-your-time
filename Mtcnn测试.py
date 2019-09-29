from torchvision import transforms
import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from tool import utils
import time
import cv2

class Detector:
    # def __init__(self, pnet_param=r'.\param9\78\pnet.pt',rnet_param=r'.\param9\112\rnet.pt',onet_param=r'.\param9\195\onet.pt', isCuda=True):
    def __init__(self, pnet_param=r'.\param1\25\pnet.pt', rnet_param=r'.\param1\30\rnet.pt',onet_param=r'.\param1\48\onet.pt', isCuda=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isCuda = isCuda
        self.pnet = torch.load(pnet_param,map_location=self.device)
        self.rnet = torch.load(rnet_param,map_location=self.device)
        self.onet = torch.load(onet_param,map_location=self.device)
        self.image_transform = transforms.Compose([transforms.ToTensor()])
        self.cls = [0.61,0.80,0.99]
        self.nms = [0.1,0.1,0.1]
        self.pScale = 0.75

    def detect(self, image):
        t = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if len(pnet_boxes) == 0:
            return np.array([])
        t_p = time.time() - t

        t = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if len(rnet_boxes) == 0:
            return np.array([])
        t_r = time.time() - t

        t = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if len(onet_boxes) == 0:
            return np.array([])
        # print('O网络留框:',onet_boxes.shape)
        t_o = time.time() - t

        t_sum = t_p + t_r + t_o
        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_p, t_r, t_o))

        return onet_boxes

    def __pnet_detect(self, image):

        boxes = []
        img = image
        w, h = img.size
        min_side_len = min(w, h)  # 获取最小宽或者高，原因：做图像金字塔；
        scale = 1

        while min_side_len > 12:
            img_data = self.image_transform(img)-0.5
            img_data = img_data.unsqueeze_(0).to(self.device)

            _cls, _offest,_ = self.pnet(img_data)
            cls, offest =_cls[0][0].cpu().data , _offest[0].cpu().data
            idxs = torch.nonzero(torch.gt(_cls[0][0].cpu().data, self.cls[0]))

            for idx in idxs:
                offest1 = offest[:,idx[0], idx[1]] #根据置信度的非0索引在置信度中取值

                boxes.append(self.__box(idx, offest1, cls[idx[0], idx[1]], scale))

            scale *= self.pScale
            _w, _h = int(w * scale), int(h * scale)
            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

        return utils.nms(np.array(boxes), self.nms[0])

    def __rnet_detect(self, image, pnet_boxes): #定义函数，包含p网络的的图片框pnet_boxes

        if len(pnet_boxes) == 0:
            return []

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)

        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.image_transform(img)-0.5
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset).to(self.device)

        _cls, _offset,_landmak = self.rnet(img_dataset)
        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        landmak = _landmak.cpu().data.numpy()


        boxes = []
        idxs, _ = np.where(cls > self.cls[1])

        for idx in idxs:
            _box = _pnet_boxes[idx] #这里把新生成的正方形中的元素，按照索引取出来，定义新的坐标
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1 #宽高
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0] #通过建议框，偏移量，计算出在原图上的坐标点
            y1 = _y1 + oh * offset[idx][1] #这里用h来算偏移量，也可以，因为现在是正方形，边长都一样；
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            x3 = _x1 + ow * landmak[idx][0]
            y3 = _y1 + oh * landmak[idx][1]
            x4 = _x2 + ow * landmak[idx][2]
            y4 = _y1 + oh * landmak[idx][3]
            x5 = _x1 + ow * landmak[idx][4]
            y5 = _y1 + oh * landmak[idx][5]
            x6 = _x1 + ow * landmak[idx][6]
            y6 = _y2 + oh * landmak[idx][7]
            x7 = _x2 + ow * landmak[idx][8]
            y7 = _y2 + oh * landmak[idx][9]

            boxes.append([x1, y1, x2, y2, x3, y3, x4, y4,x5, y5, x6, y6,x7, y7, cls[idx][0]])  # 这里把置信度加进去，for循环前要用

        return utils.nms(np.array(boxes), self.nms[1])

    def __onet_detect(self, image, rnet_boxes):

        if len(rnet_boxes) == 0:
            return []

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes) #把R网络中的图像生成新的正方形
        for _box in _rnet_boxes: #取出坐标
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.image_transform(img)-0.5
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        img_dataset = img_dataset.to(self.device)

        _cls, _offset,_landmak = self.onet(img_dataset)

        cls = _cls.cpu().data.numpy()
        #print('O网络置信度：',cls)
        offset = _offset.cpu().data.numpy()
        landmak = _landmak.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > self.cls[2])
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0] #把原图上对应框画出来
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            x3 = _x1 + ow * landmak[idx][0]
            y3 = _y1 + oh * landmak[idx][1]
            x4 = _x2 + ow * landmak[idx][2]
            y4 = _y1 + oh * landmak[idx][3]
            x5 = _x1 + ow * landmak[idx][4]
            y5 = _y1 + oh * landmak[idx][5]
            x6 = _x1 + ow * landmak[idx][6]
            y6 = _y2 + oh * landmak[idx][7]
            x7 = _x2 + ow * landmak[idx][8]
            y7 = _y2 + oh * landmak[idx][9]


            boxes.append([x1, y1, x2, y2, x3, y3, x4, y4,x5, y5, x6, y6,x7, y7, cls[idx][0]])
        return utils.nms(np.array(boxes), self.nms[2], isMin=True) #isMin=True这里去掉当大框覆盖小框的情况；

    # 将回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        #start_index特征图索引，scale缩放比例，步长为2，卷积核边长12；

        _x1 = float((start_index[1] * stride)) / scale #原图左上角点
        _y1 = float((start_index[0] * stride)) / scale
        _x2 = float((start_index[1] * stride+side_len)) / scale #原图右下角点
        _y2 = float((start_index[0] * stride+side_len)) / scale
        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[0] #反算p网络到原图的框
        y1 = _y1 + oh * offset[1]
        x2 = _x2 + ow * offset[2]
        y2 = _y2 + oh * offset[3]

        return [x1, y1, x2, y2, cls] #, x3, y3, x4, y4,x5, y5, x6, y6,x7, y7

    def boxDetect(self, inputBoxes, cons, offsets, landMark, conMax, nmsMax=0.3, iouMode='inter'):
        mask = cons > conMax
        mask_index = mask.nonzero()[:,0]

        cons = cons[mask]
        offsets = torch.index_select(offsets,dim=0,index=mask_index)
        boxes = torch.index_select(inputBoxes,dim=0,index=mask_index)
        landMark = torch.index_select(landMark,dim=0,index=mask_index)
        if cons.size(0) == 0:
            return []

        #筛选R网络的框
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]

        x1 = boxes[:,0] + w * offsets[:,0]
        y1 = boxes[:,1] + h * offsets[:,1]
        x2 = boxes[:,2] + w * offsets[:,2]
        y2 = boxes[:,3] + h * offsets[:,3]

        x3 = boxes[:, 0] + w * landMark[:, 0]
        y3 = boxes[:, 1] + h * landMark[:, 1]
        x4 = boxes[:, 2] + w * landMark[:, 2]
        y4 = boxes[:, 1] + h * landMark[:, 3]
        x5 = boxes[:, 0] + w * landMark[:, 4]
        y5 = boxes[:, 1] + h * landMark[:, 5]
        x6 = boxes[:, 0] + w * landMark[:, 6]
        y6 = boxes[:, 1] + h * landMark[:, 7]
        x7 = boxes[:, 2] + w * landMark[:, 8]
        y7 = boxes[:, 3] + h * landMark[:, 9]

        return utils.nms(torch.stack([x1, y1, x2, y2, cons, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7], dim=1), thresh=nmsMax, mode=iouMode)

    def returnBox(self, startIndex, offset, con, scale, stride=2, sideLen=12):

        _x1 = (startIndex[:,1].float() * stride)/scale
        _y1 = (startIndex[:,0].float() * stride)/scale
        _x2 = (startIndex[:,1].float() * stride + sideLen)/scale
        _y2 = (startIndex[:,0].float() * stride + sideLen)/scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[0][startIndex[:,0],startIndex[:,1]]
        y1 = _y1 + oh * offset[1][startIndex[:,0],startIndex[:,1]]
        x2 = _x2 + ow * offset[2][startIndex[:,0],startIndex[:,1]]
        y2 = _y2 + oh * offset[3][startIndex[:,0],startIndex[:,1]]

        return torch.stack([x1, y1, x2, y2, con[startIndex[:,0],startIndex[:,1]]],dim=1)


# if __name__ == '__main__':
#     detector = Detector()
#
#     path = r'E:\PycharmProjects\手写数据集\合照'
#     # path = r'E:\PycharmProjects\手写数据集\测试'
#     # path = r'E:\PycharmProjects\手写数据集\单人'
#     imagelist = os.listdir(path)
#
#     for J in imagelist:
#         with Image.open(os.path.join(path,J)) as img:
#
#             boxes = detector.detect(img)
#             imDraw = ImageDraw.Draw(img)
#             red = (255,0,0)
#
#             for box in boxes:
#                 x1 = int(box[0])
#                 y1 = int(box[1])
#                 x2 = int(box[2])
#                 y2 = int(box[3])
#                 con = float("%.4f" % box[14])  # "%.4f" % 留多少位小数
#                 """框"""
#                 imDraw.rectangle((x1, y1, x2, y2), outline=red, width=2)
#                 imDraw.text((x1, y1 - 10), str(con), fill='white')
#                 """关键点"""
#                 i = 4
#                 while i < 14:
#                     position = (int(box[i]), int(box[i + 1]), int(box[i] + 2), int(box[i + 1] + 2))
#                     i += 2
#                     imDraw.rectangle(position, outline='red')
#
#             img.show()

if __name__ == '__main__':
    cap = cv2.VideoCapture(r'E:\PycharmProjects\手写数据集\视频\1.mp4')
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    """保存视频"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter(r'E:\PycharmProjects\手写数据集\视频效果\5.avi', fourcc, fps, (width, height))  # 保存,格式avi可以,mp4不行

    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            detector = Detector()
            new_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = detector.detect(new_img)
            for box in boxes:
                x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = int(box[4]), int(box[5]), int(box[6]), int(box[7]), int(
                    box[8]), int(
                    box[9]), int(box[10]), int(box[11]), int(box[12]), int(box[13])
                # print(k1, k2, k3, k4, k5, k6, k7, k8, k9, k10)
                """画点"""
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=1)
                cv2.putText(img, "cls: {}".format("%.4f" % box[14]), (x_min, y_min-10), cv2.FONT_HERSHEY_PLAIN, 0.8,
                            color=(0, 255, 0))
                points_list = [(k1, k2), (k3, k4), (k5, k6), (k7, k8), (k9, k10)]
                for point in points_list:
                    cv2.circle(img, point, 1, color=(0, 0, 255), thickness=1)

            cv2.imshow('Frame', img)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break

        outVideo.write(img)
    cap.release()
    cv2.destroyAllWindows()




