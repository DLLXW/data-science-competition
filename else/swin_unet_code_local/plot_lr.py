import cv2
import os
import glob
from tqdm import tqdm
import torch
from torchvision import models
import matplotlib.pyplot as plt
#画出余弦学习率的变化规律
def visulize_cosine_lr(net,max_epoch,optimizer,lr_scheduler,iters=100):
    plt.figure()
    cur_lr_list = []
    cur_lr = optimizer.param_groups[-1]['lr']
    cur_lr_list.append(cur_lr)
    for epoch in range(max_epoch):
        #print('epoch_{}'.format(epoch))
        # cur_lr = optimizer.param_groups[-1]['lr']
        # cur_lr_list.append(cur_lr)
        for batch in range(iters):
            optimizer.step()
            scheduler.step(epoch + batch / iters)
        cur_lr = optimizer.param_groups[-1]['lr']
        cur_lr_list.append(cur_lr)
            #scheduler.step(epoch + batch / iters)
            #scheduler.step()
        #print('cur_lr:',cur_lr)
        #print('epoch_{}_end'.format(epoch))
        lr_scheduler.step()
        print('epoch: {},cur_lr: {}'.format(epoch,cur_lr))
    x_list = list(range(len(cur_lr_list)))
    plt.title('Cosine lr  T_0:{}  T_mult:{}'.format(T_0,T_mult))
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.plot(x_list, cur_lr_list)
    plt.savefig('./lr.png')
if __name__=='__main__':
    model=models.resnet18(pretrained=False)
    T_0=3
    T_mult=2
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-4)
    #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6, last_epoch=-1)
    visulize_cosine_lr(model,100,optimizer,scheduler,100)