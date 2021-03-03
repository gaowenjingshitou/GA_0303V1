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
from load_gestational_age import gestational_age
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


batchsz = 10
lr = 1e-1
epochs = 10
device = torch.device('cuda')
torch.manual_seed(1234)


train_db =gestational_age('/workspace/maskcnn/Mask_RCNN/datasets', 224, 'train')
val_db = gestational_age('/workspace/maskcnn/Mask_RCNN/datasets', 224, mode='val')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=1)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=1)

viz = visdom.Visdom()


def main():
    # model = ResNet18(5).to(device)
    trained_model = resnest50(pretrained=True)
    # trained_model =resnet101(pretrained=True)
    # print('children', *list(trained_model.children())[:-1])

    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]   [32,2048]
                          #                           list(trained_model.children())[-2].view(32, 2048,1,1),
                          #nn.BatchNorm2d(2048),
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          #torch.nn.Dropout(0.3),

                          nn.Linear(2048, 1000),
                          nn.Linear(2048, 1)
                          # nn.ReLU(inplace=True),
                          # torch.nn.Dropout(0.5),
                          # nn.Linear(1024, 1),
                          #nn.ReLU(inplace=True)
                          #nn.ReLU6(inplace=True)

                          ).to(device)

    model.train()

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.99,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    #     criteon = nn.CrossEopyLoss()
    criteon = torch.nn.MSELoss(reduce=True, size_average=True)
    criteon=nn.SmoothL1Loss(size_average=True, reduce=True, reduction='mean')
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
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='loss_val', opts=dict(title='loss_val'))
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
                    scheduler.step(loss_val)
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

#     train_loader1 = DataLoader(train_db, batch_size=140, shuffle=True,
#                                num_workers=0)
#     val_loader1 = DataLoader(val_db, batch_size=60, num_workers=0)
#     for x, y in train_loader1:
#         criteon = torch.nn.MSELoss()
#         x, y = x.to(device), y.to(device)
#         print(x.shape)
#         with torch.no_grad():
#             logits_train = model(x)
#             # print('logit_train',logits_train)
#             logits_train = logits_train.squeeze(1)
#             logits_train = logits_train.cpu().numpy()
#             y = y.cpu().numpy()
#             r2 = r2_score(y, logits_train)
#             print("r2", r2)
#             mse = mean_squared_error(y, logits_train)
#             # loss = criteon(y, logits_train)
#             # print("loss",loss)
#             print("mse", mse)
#             print('logits_train', logits_train)
#             # logits_train=pd.DataFrame(logits_train)
#             print('train_y', y)

#     for x, y in val_loader1:
#         x, y = x.to(device), y.to(device)
#         print(x.shape)
#         with torch.no_grad():
#             logits_val = model(x)
#             logits_val = logits_val.squeeze(1)
#             logits_val = logits_val.cpu().numpy()
#             y = y.cpu().numpy()
#             r2 = r2_score(y, logits_val)
#             print("r2", r2)
#             mse = mean_squared_error(y, logits_val)
#             print("mse", mse)
#             print('logits_val', logits_val)
            # print('logits_train',logits_train)
            # logits_train=pd.DataFrame(logits_train)
            #print('test_y', y)
#     plt.plot(epoch, train_loss1,'go-')
#     plt.ylabel('Loss')
#     plt.xlabel('epoch')
#     plt.plot(epoch, val_loss1, 'rs')
#     plt.show()
#     test_acc = evalute(model, test_loader)
#     print('test acc:', test_acc)

if __name__ == '__main__':
    main()