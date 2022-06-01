
from re import A
from turtle import towards
import torch
from torch import nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import random
import cv2
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import timm
import torch.nn.functional as F
from model_triplet import DogNet
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import csv

class DogDataset(Dataset):
    def __init__(self, df,data_root,
                 transform=None):
        super(Dataset, self).__init__()
        self.df = df
        self.transform=transform
        self.data_root=data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        imgA=cv2.imread(os.path.join(self.data_root,row['imageA']))
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgB=cv2.imread(os.path.join(self.data_root,row['imageB']))
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
        #
        imgA = self.transform(image=np.array(imgA))['image']
        imgB = self.transform(image=np.array(imgB))['image']
        #
        return imgA,imgB
def get_inference_transforms(input_size):
    return A.Compose([
            A.Resize(input_size,input_size),#[height,width] 
            #A.HorizontalFlip(p=0.5),
            #A.RandomBrightnessContrast(p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            ),
        ToTensorV2(p=1.0)
    ])

@torch.no_grad()
def inference(model, data_loader, device):
    model.eval()
    image_preds_all = []
    img_path_all=[]
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgA,imgB) in pbar:
        imgA=imgA.to(device).float()
        imgB=imgB.to(device).float()
        dis_pos,_,_,_,_  = model(imgA,imgB,imgB)
        #print(image_preds)
        image_preds=1/dis_pos.detach().cpu().numpy() 
        image_preds_all+=image_preds.tolist()
    #
    return np.array(image_preds_all)
if __name__=="__main__":
    data_root='/home/yangdeyu/competition/qyl/data/pet_biometric_challenge_2022/test/test'
    df = pd.read_csv('/home/yangdeyu/competition/qyl/data/pet_biometric_challenge_2022/test/test_data.csv')
    device = torch.device('cuda')
    save_dir='submit'
    os.makedirs(save_dir,exist_ok=True)
    preds_lst=[]
    for tta in range(1):
        infer_ds = DogDataset(df,data_root,get_inference_transforms(224))
        infer_loader = torch.utils.data.DataLoader(
                infer_ds,
                batch_size=8,
                num_workers=4,
                shuffle=False,
                pin_memory=False,
            )
        # for fold in range(5):
        #     print('Inference TTA:{} fold {} started'.format(tta,fold))
        #     model = ImgClassifier('convnext_large_in22ft1k', 21, pretrained=False).to(device)
        #     model.load_state_dict(torch.load('./ckpt/convnext_large_in22ft1k/convnext_large_in22ft1k_fold_{}_best.pth'.format(fold)))
        #     preds_lst.append(inference(model, infer_loader, device))
        #     del model
        #     torch.cuda.empty_cache()
        for fold in range(3):
            print('Inference TTA:{} fold {} started'.format(tta,fold))
            model = DogNet('convnext_small', pretrained=False).to(device)
            model.load_state_dict(torch.load('./ckpt_0520_triplet/convnext_small/convnext_small_fold_{}_best.pth'.format(fold)))
            preds_lst.append(inference(model, infer_loader, device))
            del model
            torch.cuda.empty_cache()
    #模型融合
    print(len(preds_lst))
    #ratio=[0.6/5]*5+[0.4/5]*5
    ratio=[1/len(preds_lst)]*len(preds_lst)
    assert abs(np.sum(ratio)-1)<1e-3
    for i in range(len(preds_lst)):
        if i==0:
            preds=ratio[i]*preds_lst[i]
        else:
            preds+=ratio[i]*preds_lst[i]
    #
    #print(preds,len(preds))
    df['prediction']=preds
    print(df)
    df.to_csv(save_dir+'/trituge_528.csv',index=False)