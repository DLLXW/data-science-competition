

import cv2
from PIL import Image
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset,DataLoader
import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2

def get_train_transforms(height_size,width_size):
    return A.Compose([
            A.Resize(height_size,width_size),#[height,width]
            #A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.ShiftScaleRotate(rotate_limit=10,p=0.5),
            #A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.OneOf([
                    # 模糊相关操作
                    A.MotionBlur(p=.75),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.Blur(blur_limit=3, p=0.75),
                ], p=0.5),
            A.OneOf(
                [
                A.CoarseDropout(max_holes=8,
                            max_height=8,
                            max_width=8,
                            p=0.5),
                A.Cutout(
                    num_holes=8,
                    max_h_size=8,
                    max_w_size=8,
                    p=0.5,)
                    ],p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms(height_size,width_size):
    return A.Compose([
            A.Resize(height_size,width_size),#
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
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


if __name__=="__main__":
    train_csv = pd.read_csv('../train_df.csv')
    img=get_img('../../input/phase2_train/img_4.jpg')
    cv2.imwrite('aug_before.jpg',img)
    aug=A.Rotate(limit=[180,180],p=1.)
    aug_img=aug(image=img)['image']
    cv2.imwrite('aug_after.jpg',aug_img)