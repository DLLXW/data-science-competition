import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn 
import torch.nn.functional as F 

import engine
from dataset import DogDataset
from model import DogModel
from config import CFG
from custom_scheduler import DogScheduler
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
DATA_DIR = '../data/pet_biometric_challenge_2022/train/images/'
TRAIN_CSV = '../data/pet_biometric_challenge_2022/train/train_data.csv'
MODEL_PATH = '.'
os.makedirs(MODEL_PATH,exist_ok=True)
def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])

def get_valid_transforms(img_size=512):

    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])
#
def run_training():
    df = pd.read_csv(TRAIN_CSV)
    # labelencoder= LabelEncoder()
    # df['label_group'] = labelencoder.fit_transform(df['label_group'])
    trainset = DogDataset(df,
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
    model = DogModel()
    model.to(CFG.device)
    #model=torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().to(CFG.device) 

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr = CFG.scheduler_params['lr_start'])
    scheduler = DogScheduler(optimizer, **CFG.scheduler_params)

    for epoch in range(CFG.epochs):
        avg_loss_train = engine.train_fn(model, trainloader, optimizer, scheduler, epoch, CFG.device,criterion,scaler)
        #avg_loss_eval=engine.eval_fn(model, valloader, epoch, device,criterion,scaler)
        torch.save(model.state_dict(), MODEL_PATH + 'arcface_224x224_{}.pt'.format(CFG.model_name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            },
            MODEL_PATH + 'arcface_224x224_{}_checkpoints.pt'.format(CFG.model_name)
        )

if __name__=='__main__':
    run_training()