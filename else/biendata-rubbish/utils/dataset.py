
from albumentations.augmentations.transforms import RandomCrop
import cv2
from PIL import Image
import torchvision
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
import albumentations as A

from albumentations.pytorch import ToTensorV2

def get_train_transforms(input_size):
    return Compose([
            Resize(input_size,input_size),#[height,width] 
            #RandomResizedCrop(input_size, input_size,scale=(0.5, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.25),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.25),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms(input_size):
    return Compose([
            Resize(input_size,input_size),#276, 344
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_img(path):
    #im_bgr = cv2.imread(path)
    #im_rgb = im_bgr[:, :, ::-1]
    im_rgb=np.array(Image.open(path))
    if len(im_rgb.shape)==2:
        #im_rgb=np.stack([im_rgb,im_rgb,im_rgb],axis=2)
        im_rgb=np.expand_dims(im_rgb,axis=-1)
    return im_rgb
def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class RubbishDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 input_size=384,
                 in_channels=3,
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.input_size=input_size
        self.data_root = data_root
        self.in_channels=in_channels#输入通道数量
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['class_id'].values
            #print(self.labels)
            if one_hot_label is True:
                self.labels = np.eye(self.df['class_id'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
        #print("{}/{}".format(self.data_root, self.df.loc[index]['file_name']))
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['file_name']))
        #print(img.shape)
        if self.transforms:
            img = self.transforms(image=img)['image']
        if self.output_label == True:
            return img,target
        else:
            return img

if __name__=="__main__":
    data_root='../data/train'
    train_csv = pd.read_csv('../data/label_class.csv')
    train_ = train_csv
    train_ds = RubbishDataset(train_, data_root, transforms=get_train_transforms(384), output_label=True,input_size=384, )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=0,
        #sampler=BalanceClassSampler(labels=train_['class_id'].values, mode="downsampling")
    )
    for step,(imgs, image_labels) in enumerate(train_loader):
        #[c,h,w]->[h,w,c]
        img=imgs[0].permute(1,2,0)
        print(img.shape,image_labels)
        print(isinstance(image_labels,list))
        cv2.imwrite('aug.jpg',img.numpy())
        break