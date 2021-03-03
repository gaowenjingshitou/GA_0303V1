import  torch
import  os, glob
import  random, csv
from    torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter
# from    resnet import ResNet18
from torchvision.models import resnet101
from sklearn.metrics import confusion_matrix
# from util import Flatten
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


class HC(Dataset):

    def __init__(self, root, resize, mode):
        super(HC, self).__init__()

        self.root = root
        self.resize = resize

        #         self.name2label = {} # "sq...":0
        #         for name in sorted(os.listdir(os.path.join(root))):
        #             if not os.path.isdir(os.path.join(root, name)):
        #                 continue

        #             self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)

        # image, labels
        self.names, self.images, self.labels = self.load_csv('HC_TRAIN.csv')

        if mode == 'train':  # 60%
            self.images = self.images[:int(0.7 * len(self.images))]
            self.labels = self.labels[:int(0.7 * len(self.labels))]
        #         elif mode=='val': # 20% = 60%->80%
        #             self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
        #             self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        elif mode == 'val':  # 20% = 80%->100%
            self.images = self.images[int(0.7 * len(self.images)):]
            self.labels = self.labels[int(0.7 * len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            # for name in self.name2label.keys():
            # 'pokemon\\mewtwo\\00001.png
            #                 images += glob.glob(os.path.join(self.root, name, '*.png'))
            # images=glob.glob(os.path.join('/workspace/maskcnn/Mask_RCNN/datasets' , 'result_segementation', '*.jpg'))  # '/workspace/maskcnn/Mask_RCNN/datasets/result_segementation/0004866459_917_0535.jpg',
            images += glob.glob(os.path.join(self.root, '*.png'))
            # images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # 1167, 'pokemon\\bulbasaur\\00000000.png'

            random.shuffle(images)
            # print(images)
            #             Images=[image_path.split("/")[-1].split("_")[0] for image_path in images]   #'0005163074',

            # print(len(rImages), Images)

            Total_EXAM_NO = pd.read_csv("/workspace/HC18_data/training_set_pixel_size_and_HC.csv", index_col=0)
            Total_EXAM_NO = Total_EXAM_NO['head circumference (mm)']
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:

                writer = csv.writer(f)
                for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split("/")[-1]
                    # print("name",name)
                    if name in Total_EXAM_NO.index:
                        print(name)
                        # name = img.split(os.sep)[-1]
                        label = Total_EXAM_NO.loc[name]
                        print("label", label)

                        # label = self.name2label[name]
                        # 'pokemon\\bulbasaur\\00000000.png', 0
                        writer.writerow([name, img, label])
                print('writen into csv file:', filename)

            # read from csv file

        names, images, labels = [], [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                name, img, label = row
                label = float(label)
                names.append(name)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return names, images, labels

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
        name, img, label = self.names[idx], self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),

            #              transforms.RandomAffine((-180, 180), translate=None, scale=None, shear=None, resample=False, fillcolor=0),
            #              transforms.ColorJitter(brightness=50, contrast=0, saturation=0, hue=0),
            #             transforms.RandomHorizontalFlip(p=0.5),
            #             transforms.RandomVerticalFlip(p=0.5),
            #             transforms.RandomCrop(self.resize) , #+
            #             transforms.RandomResizedCrop(self.resize) ,#+

            #             transforms.RandomHorizontalFlip(p=0.5),
            #             transforms.RandomVerticalFlip(p=0.5),

            # #             transforms.ToPILImage(),

            #             transforms.RandomRotation((30,60), resample=False, expand=False, center=None),
            #             transforms.ColorJitter(brightness=50, contrast=0, saturation=0, hue=0),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return name, img, label


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

    db = HC('/workspace/HC18_data/training_set', 224, 'train')

    name, x, y = next(iter(db))
    print("name", name, 'sample:', x.shape, y.shape, y)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=0)

    for name, x, y in loader:
        print('name', name)
        print("batch sample", x.shape, y.shape)

        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(name), win='image_name', opts=dict(title='batch-name'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)


if __name__ == '__main__':
    main()


