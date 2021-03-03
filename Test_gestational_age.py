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
import pandas as pd
# from load_gestational_age import gestational_age
#from load_gestational_agev2 import gestational_age
from  load_gestational_agev3 import gestational_age
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


batchsz = 32
lr = 1e-4
epochs = 1000
device = torch.device('cuda')
torch.manual_seed(1234)
train_db =gestational_age('/workspace/maskcnn/Mask_RCNN/datasets', 224, 'train')
val_db = gestational_age('/workspace/maskcnn/Mask_RCNN/datasets', 224, mode='val')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=1)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=1)


def evalute(model, loader):
    model.eval()

    names, y_trues, y_preds = [], [], []

    total = len(loader.dataset)

    for step, (name, x, y) in enumerate(val_loader):
        print("Evaluate {} val test".format(step))
        names.extend(name)
        x, y = x.to(device), y.to(device)
        y = y.cpu().numpy()
        y = list(y)
        #         print('y',y)
        y_trues.extend(y)

        with torch.no_grad():
            logits = model(x)
            logits = logits.squeeze()
            logits = logits.cpu().numpy()
            logits = list(logits)
            #             print('y_pred',logits)
            y_preds.extend(logits)
    print("y_trues", y_trues)
    print("y_preds", y_preds)

    return names, y_trues, y_preds

    # pred = logits.argmax(dim=1)


def main():
    # model = ResNet18(5).to(device)
    # trained_model = resnest50(pretrained=True)
    # trained_model =resnet101(pretrained=True)
    # print('children', *list(trained_model.children())[:-1])
    trained_model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]   [32,2048]
                          #                           list(trained_model.children())[-2].view(32, 2048,1,1),
                          # nn.BatchNorm2d(2048),
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          # torch.nn.Dropout(0.3),

                          #                           nn.Linear(2048, 1000),
                          #                           nn.Dropout(0.1),
                          #                           nn.Linear(1000,500),
                          #                           nn.Dropout(0.1),
                          #                           nn.Linear(500, 1)
                          # nn.ReLU(inplace=True),
                          # torch.nn.Dropout(0.5),
                          # nn.Linear(1024, 1),
                          # nn.ReLU(inplace=True)
                          # nn.ReLU6(inplace=True)
                          nn.Linear(2048, 1),
                          #                         nn.Linear(1000, 2000),

                          #                           nn.Linear(2000, 1),
                          # nn.LeakyReLU(inplace=True),
                          ).to(device)
    model.load_state_dict(torch.load('best06301936v0.mdl'))

    print('loaded from ckpt!')

    names, val_y_true, val_preds = evalute(model, val_loader)
    #     names=list(names)
    #     val_y_true=list(val_y_true)
    #     val_preds=list(val_preds)
    from pandas.core.frame import DataFrame

    c = {"names": names, "val_y_true": val_y_true, "val_preds": val_preds}  # 将列表a，b转换成字典
    data = DataFrame(c)  # 将字典转换成为数据框
    print(data)
    # a.to_excel("val_values07011427.xls")
    # data.to_excel("val_values07020812.xls")
    # data.to_excel("val_values07020812.xls")
    # data.to_excel("val_values07021000.xls")  #最好的结果
    data.to_excel("val_values07061911.xls")


if __name__ == '__main__':
    main()
