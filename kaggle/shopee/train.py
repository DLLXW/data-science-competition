import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('./utils-shopee')
sys.path.append('./pytorch-image-models-master')
#
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import timm
import torch
from torch import nn 
import torch.nn.functional as F 

import engine
from dataset import ShopeeDataset
from custom_scheduler import ShopeeScheduler
from augmentations import get_train_transforms
from model import ShopeeModel
from config import CFG

from torch.cuda.amp import autocast, GradScaler
DATA_DIR = './data/train_images'
TRAIN_CSV = './utils-shopee/folds.csv'
MODEL_PATH = './'
#
'''
If you are using kaggle GPU, you must have to change batch_size. In addition, you also have to change CFG.lr_max = 1e-5 * 32
'''
#
def run_training():
    
    df = pd.read_csv(TRAIN_CSV)

    labelencoder= LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])

    trainset = ShopeeDataset(df,
                             DATA_DIR,
                             transform = get_train_transforms(img_size = CFG.img_size))

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = CFG.batch_size,
        num_workers = CFG.num_workers,
        #pin_memory = True,
        shuffle = True,
        #drop_last = True
    )
    scaler=GradScaler()
    model = ShopeeModel()
    model.to(CFG.device)
    #model=torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().to(CFG.device) 

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = CFG.scheduler_params['lr_start'])
    scheduler = ShopeeScheduler(optimizer, **CFG.scheduler_params)

    for epoch in range(CFG.epochs):
        avg_loss_train = engine.train_fn(model, trainloader, optimizer, scheduler, epoch, CFG.device,criterion,scaler)
        torch.save(model.state_dict(), MODEL_PATH + 'arcface_512x512_{}.pt'.format(CFG.model_name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            },
            MODEL_PATH + 'arcface_512x512_{}_checkpoints.pt'.format(CFG.model_name)
        )

if __name__=='__main__':
    run_training()