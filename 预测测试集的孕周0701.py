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
# from load_gestational_age import gestational_age
from load_gestational_agev2 import gestational_age
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
                      nn.Linear(2048, 1000),
                      nn.Linear(1000, 500),

                      nn.Linear(500, 1)
                      # nn.LeakyReLu(inplace=True)
                      ).to(device)

model.load_state_dict(torch.load('best06301936.mdl'))

print('loaded from ckpt!')


def evalute(model, loader):
    model.eval()

    #     val_preds= []
    #     val_y_true=[]

    total = len(loader.dataset)

    y_trues = [], y_preds = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y = y.cpu().numpy()
        y_trues.append(y)

        with torch.no_grad():
            logits = model(x)
            logits = logits.cpu().numpy()
            # val_preds+=logits
            y_preds.append(logits)

    return y_trues, val_preds

    # pred = logits.argmax(dim=1)

    # pred = logits.argmax(dim=1)


#train_loader1 = DataLoader(train_db, batch_size=1310, shuffle=True,
                            #   num_workers=0)
val_db = gestational_age('/workspace/maskcnn/Mask_RCNN/datasets', 224, mode='val')
val_loader1 = DataLoader(val_db, batch_size=32, num_workers=0)

val_values= evalute(model, val_loader1)

print("Y_true",val_values[0])
print("val_preds",val_values[2])