import torch
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader

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
#,index_col="EXAM_NO"


batchsz = 32
lr = 3e-4
epochs = 500
device = torch.device('cuda')
torch.manual_seed(1234)


Total_EXAM_NO=pd.read_csv("Gestational_age_路径索引1.csv",converters={'EXAM_NO':str},header=None,names=['Image_path','Gestational_week'])


Image_path1=[image_path.split("/")[-1].split("_")[0] for image_path in Total_EXAM_NO['Image_path']]
Total_EXAM_NO['Image_path']=Image_path1

Total_EXAM_NO=Total_EXAM_NO.set_index(Total_EXAM_NO['Image_path'])
Total_EXAM_NO.drop("Image_path",axis=1,inplace=True)


class Data(Dataset):

    def __init__(self, root, resize, mode):
        super(Data, self).__init__()

        self.root = root
        self.resize = resize

        self.images, self.labels = self.load_csv('Gestational_age_路径索引1.csv')

        if mode == 'train':  # 60%
            self.images = self.images[:int(0.7 * len(self.images))]
            self.labels = self.labels[:int(0.7 * len(self.labels))]
        elif mode == 'val':  # 20% = 60%->80%
            self.images = self.images[int(0.3 * len(self.images)):]
            self.labels = self.labels[int(0.3 * len(self.labels)):]

    def load_csv(self, filename):

        #         if not os.path.exists(os.path.join( filename)):
        #             images = []
        #             for name in self.name2label.keys():
        #                 # 'pokemon\\mewtwo\\00001.png
        #                 images += glob.glob(os.path.join(self.root, name, '*.png'))
        #                 images += glob.glob(os.path.join(self.root, name, '*.jpg'))
        #                 images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

        #             # 1167, 'pokemon\\bulbasaur\\00000000.png'
        #             print(len(images), images)

        #             random.shuffle(images)
        #             with open(os.path.join(filename), mode='w', newline='') as f:
        #                 writer = csv.writer(f)
        #                 for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
        #                     name = img.split(os.sep)[-2]
        #                     label = self.name2label[name]
        #                     # 'pokemon\\bulbasaur\\00000000.png', 0
        #                     writer.writerow([img, label])
        #                 print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row

                images.append(img)
                labels.append(label)

        labels = [float(x) for x in labels]

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
            lambda x: x.filter(ImageFilter.SMOOTH),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            transforms.Resize((int(self.resize * 1.3), int(self.resize * 1.3))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            # .RandomCrop
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label

train_db = Data('./result1', 224, mode='train')
val_db = Data('./result1', 224, mode='val')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=1)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=1)


def load_csv(filename):
    # read from csv file
    images, labels = [], []
    with open(os.path.join(filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 'pokemon\\bulbasaur\\00000000.png', 0
            img, label = row

            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)

    return images, labels

images,labels=load_csv("Gestational_age_路径索引1.csv")



viz = visdom.Visdom()




def main():
    # model = ResNet18(5).to(device)
    trained_model = resnest50(pretrained=True)
    # trained_model =resnet101(pretrained=True)
    # print('children', *list(trained_model.children())[:-1])

    model = nn.Sequential(*list(trained_model.children())[:-2],  # [b, 512, 1, 1]   [32,2048]
                          #                           list(trained_model.children())[-2].view(32, 2048,1,1),
                          nn.BatchNorm2d(2048),
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          # torch.nn.Dropout(0.5),

                          nn.Linear(2048 * 49, 1),
                          # nn.ReLU(inplace=True),
                          # torch.nn.Dropout(0.5),
                          # nn.Linear(1024, 1),
                          nn.ReLU(inplace=True)
                          ).to(device)

    model.train()

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.99,weight_decay=0.01)
    # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    #     criteon = nn.CrossEopyLoss()
    criteon = torch.nn.MSELoss()

    # criteon = nn.L1Loss()

    #     for t in range(200):
    #     prediction = model(x)
    #     loss = loss_func(prediction, y)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if t%5 == 0:
    #         plt.cla()
    #         plt.scatter(x.data.numpy(), y.data.numpy()) # 画散点图
    #         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5) # 画拟合曲线
    #         plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size':20,'color':'red'}) # 显示损失数值
    #         plt.pause(0.1)

    # # 如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。要想防止这种情况，需要在plt.show()之前加上ioff()命令。
    # plt.ioff()
    # plt.show()

    best_loss, best_epoch = 5, 0
    global_step = 0
    viz.line([1], [-1], win='loss', opts=dict(title='loss'))
    viz.line([10], [-1], win='loss_val', opts=dict(title='loss_val'))
    #     train_loss1=[]
    #     val_loss1=[]

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            # print("Enter"+str(epoch)+'time'+str(step))
            print("Enter: {} step {}".format(epoch, step))

            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            # train_loss1.append(loss)

            # loss = criteon(logits, y)
            # loss=F.smooth_l1_loss(logits, y,size_average=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:

            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    logits_val = model(x)

                    #                     print('logits_val',logits_val)
                    #                     print("y_true",y)
                    loss_val = criteon(logits_val, y)
                    # val_loss1.append(loss_val)
                    # scheduler.step(loss_val)
                    #                     r2=r2_score(logits_val, y)
                    #                     mean_absolute_error=mean_absolute_error(logits_val, y)
                    #                     mean_squared_error=mean_squared_error(logits_val, y)
                    #                     print("r2",r2)
                    #                     print("mean_absolute_error",mean_absolute_error)
                    #                     print("mean_squared_error",mean_squared_error)
                    # loss_val = criteon(logits_val, y)
                    # loss_val =F.smooth_l1_loss(logits_val, y,size_average=False)

                    if loss_val.item() < best_loss:
                        best_epoch = epoch
                        best_loss = loss_val

                        torch.save(model.state_dict(), 'best.mdl')

                        viz.line([loss_val.item()], [global_step], win='loss_val', update='append')

    print('best loss:', best_loss, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')
    # train_db = Data('./result1', 224, mode='train')
    # val_db = Data('./result1', 224, mode='val')

    train_loader1 = DataLoader(train_db, batch_size=140, shuffle=True,
                               num_workers=0)
    val_loader1 = DataLoader(val_db, batch_size=60, num_workers=0)
    for x, y in train_loader1:
        criteon = torch.nn.MSELoss()
        x, y = x.to(device), y.to(device)
        print(x.shape)
        with torch.no_grad():
            logits_train = model(x)
            # print('logit_train',logits_train)
            logits_train = logits_train.squeeze(1)
            logits_train = logits_train.cpu().numpy()
            y = y.cpu().numpy()
            r2 = r2_score(y, logits_train)
            print("r2", r2)
            mse = mean_squared_error(y, logits_train)
            # loss = criteon(y, logits_train)
            # print("loss",loss)
            print("mse", mse)
            print('logits_train', logits_train)
            # logits_train=pd.DataFrame(logits_train)
            print('train_y', y)

    for x, y in val_loader1:
        x, y = x.to(device), y.to(device)
        print(x.shape)
        with torch.no_grad():
            logits_val = model(x)
            logits_val = logits_val.squeeze(1)
            logits_val = logits_val.cpu().numpy()
            y = y.cpu().numpy()
            r2 = r2_score(y, logits_val)
            print("r2", r2)
            mse = mean_squared_error(y, logits_val)
            print("mse", mse)
            print('logits_val', logits_val)
            # print('logits_train',logits_train)
            # logits_train=pd.DataFrame(logits_train)
            print('test_y', y)
#     plt.plot(epoch, train_loss1,'go-')
#     plt.ylabel('Loss')
#     plt.xlabel('epoch')
#     plt.plot(epoch, val_loss1, 'rs')
#     plt.show()
#     test_acc = evalute(model, test_loader)
#     print('test acc:', test_acc)

#     confusion, report, AUC = test_confusion(model, test_loader)
#     print('confusion', confusion)
#     print('classification_report', report)
#     print('AUC', AUC)




if __name__ == '__main__':
    main()

