
from re import A
import torch
from torch import nn
import os
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
from utils.dataset import RubbishDataset
from net.model import ImgClassifier
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_inference_transforms(input_size):
    return Compose([
            Resize(input_size,input_size),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.25),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.25),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

@torch.no_grad()
def inference(model, data_loader, device):
    model.eval()
    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        image_preds = model(imgs)
        #print(image_preds,F.softmax(image_preds,dim=1).detach().cpu().numpy().tolist())
        #break
        image_preds_all += F.softmax(image_preds,dim=1).detach().cpu().numpy().tolist()
    #
    return image_preds_all
if __name__=="__main__":
    lable_cls={0:'A',
            1:'B',
            2:'C',
            3:'D',
            4:'E',
            5:'F'}

    preds = []
    use_kfold=True
    input_size=384
    infer_df = pd.read_csv('./data/sample_submission.csv')
    infer_df['file_name']=infer_df['id']
    device = torch.device('cuda')
    for tta in range(10):
        infer_ds_1 = RubbishDataset(infer_df, './data/validation/', transforms=get_inference_transforms(384), output_label=False)
        infer_loader_1 = torch.utils.data.DataLoader(
                infer_ds_1,
                batch_size=32,
                num_workers=4,
                shuffle=False,
                pin_memory=False,
            )
        infer_ds_2 = RubbishDataset(infer_df, './data/validation/', transforms=get_inference_transforms(512), output_label=False)
        infer_loader_2 = torch.utils.data.DataLoader(
                infer_ds_2,
                batch_size=32,
                num_workers=4,
                shuffle=False,
                pin_memory=False,
            )
        if use_kfold:
            for fold in range(5):
                print('Inference TTA:{} fold {} started'.format(tta,fold))
                model = ImgClassifier('swin_large_patch4_window12_384', 6, pretrained=False).to(device)
                model.load_state_dict(torch.load('./ckpt/swin_large_patch4_window12_384_ranger/swin_large_patch4_window12_384_fold_{}_last.pth'.format(fold)))
                preds.append(inference(model, infer_loader_1, device))
                del model
                model = ImgClassifier('tf_efficientnet_b5', 6, pretrained=False).to(device)
                model.load_state_dict(torch.load('./ckpt/tf_efficientnet_b5_ranger/tf_efficientnet_b5_fold_{}_best.pth'.format(fold)))
                preds.append(inference(model, infer_loader_1, device))
                del model
                torch.cuda.empty_cache()
        else:
            print('Inference TTA:{} all data'.format(tta))
            model = ImgClassifier('swin_large_patch4_window12_384', 6, pretrained=False).to(device)
            model.load_state_dict(torch.load('./ckpt/swin_base_patch4_window12_384/swin_base_patch4_window12_384_all_data_last.pth'))
            preds.append(inference(model, infer_loader_1, device))
            del model
            torch.cuda.empty_cache()
    print('total pre len:{}'.format(len(preds)))
    preds = np.mean(preds, axis=0)
    preds=np.argmax(preds, 1)
    print(preds)
    preds=[lable_cls[p] for p in preds]
    #print(preds)
    infer_df['label']=preds
    infer_df[['id','label']].to_csv('submit.csv',index=False)
    #print(infer_df)
    # a=[[[1,2,1,3],[1,2,1,3]],[[3,1,1,2],[1,1,1,-1]]]
    # b=np.mean(a,axis=0)
    # print(b)
    # print(np.argmax(b, axis=1))
