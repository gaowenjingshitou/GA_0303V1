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

viz = visdom.Visdom()


from sklearn import metrics
from sklearn.metrics import auc

def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total


from sklearn.metrics import roc_curve, auc


def test_confusion(model, loader):
    #
    model.eval()

    y_test = []
    predict = []
    logit_tests = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logit_test = model(x)
            print("logit_test", logit_test)
            pred = logit_test.argmax(dim=1)
            pred = pred.cpu()
            y = y.cpu()
            logit_test = logit_test.cpu().numpy()
            pred = list(pred.numpy())

            test_y = list(y.numpy())
            logit_test1 = list(logit_test[:, 1])

            y_test += test_y
            predict += pred
            logit_tests += logit_test1
    print('y_test', y_test)
    print('predict', predict)

    confusion = confusion_matrix(y_test, predict)

    report = classification_report(y_test, predict)
    fpr, tpr, threshold = roc_curve(y_test, logit_tests, pos_label=1)

    AUC = auc(fpr, tpr)  ###计算auc的值

    plt.plot(fpr, tpr)
    plt.title('ROC_curve' + '(AUC: ' + str(AUC) + ')')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("ROC.pdf")
    plt.savefig("ROC.svg")
    plt.show()

    return confusion, report, AUC


def main():
    # model = ResNet18(5).to(device)

    # trained_model = resnet101(pretrained=True)
    # # print('children', *list(trained_model.children())[:-1])
    #
    # model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
    #                       Flatten(),  # [b, 512, 1, 1] => [b, 512]
    #                       nn.Linear(2048, 2)
    #                       ).to(device)
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
                          nn.Linear(2048, 2),
                          #                         nn.Linear(1000, 2000),

                          #                           nn.Linear(2000, 1),
                          # nn.LeakyReLU(inplace=True),
                          ).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            # print("Enter"+str(epoch)+'time'+str(step))
            print("Enter: {} step {}".format(epoch, step))

            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:

            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.mdl')

                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)

    confusion, report, AUC = test_confusion(model, test_loader)
    print('confusion', confusion)
    print('classification_report', report)
    print('AUC', AUC)

if __name__ == '__main__':
    main()

