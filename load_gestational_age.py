import  torch
import  os, glob
import  random, csv

from    torch.utils.data import Dataset, DataLoader

from    torchvision import transforms
from    PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter
# from    resnet import ResNet18
from torchvision.models import resnet101
from sklearn.metrics import confusion_matrix
from util import Flatten
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import numpy as np
import matplotlib.pyplot as plt
from resnest.torch import resnest50
# net = resnest50(pretrained=True)
import torch.nn.functional as F
import pandas as pd


class gestational_age(Dataset):

    def __init__(self, root, resize, mode):
        super(gestational_age, self).__init__()

        self.root = root
        self.resize = resize

        #         self.name2label = {} # "sq...":0
        #         for name in sorted(os.listdir(os.path.join(root))):
        #             if not os.path.isdir(os.path.join(root, name)):
        #                 continue

        #             self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv('image4.csv')

        if mode == 'train':  # 60%
            self.images = self.images[:int(0.8 * len(self.images))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        #         elif mode=='val': # 20% = 60%->80%
        #             self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
        #             self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        else:  # 20% = 80%->100%
            self.images = self.images[int(0.2 * len(self.images)):]
            self.labels = self.labels[int(0.2 * len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            # for name in self.name2label.keys():
            # 'pokemon\\mewtwo\\00001.png
            #                 images += glob.glob(os.path.join(self.root, name, '*.png'))
            # images=glob.glob(os.path.join('/workspace/maskcnn/Mask_RCNN/datasets' , 'result_segementation', '*.jpg'))  # '/workspace/maskcnn/Mask_RCNN/datasets/result_segementation/0004866459_917_0535.jpg',
            images = glob.glob(os.path.join(self.root, 'result_segementation', '*.jpg'))
            # images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # 1167, 'pokemon\\bulbasaur\\00000000.png'

            random.shuffle(images)
            print(images)
            #             Images=[image_path.split("/")[-1].split("_")[0] for image_path in images]   #'0005163074',

            # print(len(Images), Images)

            Total_EXAM_NO = pd.read_csv("Gestational_age_路径索引1.csv", converters={'EXAM_NO': str}, header=None,
                                        names=['Image_path', 'Gestational_week'])
            Total_EXAM_NO = Total_EXAM_NO.set_index('Image_path')
            Image_path1 = [image_path.split("/")[-1].split("_")[0] for image_path in Total_EXAM_NO.index]
            Total_EXAM_NO.index = Image_path1

            # label=Total_EXAM_NO.loc['0005033712']
            # label=label.astype(float)
            #             label=label.values
            #             #type(label)
            #             label=label[0]
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split("/")[-1].split("_")[0]
                    # print("name",name)
                    if name in Total_EXAM_NO.index:
                        # name = img.split(os.sep)[-1]
                        label = Total_EXAM_NO.loc[name]
                        label = label.values
                        # type(label)
                        label = label[0]
                        # label = self.name2label[name]
                        # 'pokemon\\bulbasaur\\00000000.png', 0
                        writer.writerow([img, label])
                print('writen into csv file:', filename)

            # read from csv file
            images, labels = [], []
            with open(os.path.join(self.root, filename)) as f:
                reader = csv.reader(f)
                for row in reader:
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    img, label = row
                    label = float(label)

                    images.append(img)
                    labels.append(label)

            assert len(images) == len(labels)

            return images, labels

    def __len__(self):

        return len(self.images)

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label

def main():

    import visdom
    import time
    import torchvision

    viz = visdom.Visdom()

    # tf = transforms.Compose([
    #                 transforms.Resize((64,64)),
    #                 transforms.ToTensor(),
    # ])
    # db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # print(db.class_to_idx)
    #
    # for x,y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)

    db = gestational_age('/workspace/maskcnn/Mask_RCNN/datasets', 224, 'train')

    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)
    
    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=1)

    for x, y in loader:
        print("batch sample", x.shape, y.shape)
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)

if __name__ == '__main__':
    main()