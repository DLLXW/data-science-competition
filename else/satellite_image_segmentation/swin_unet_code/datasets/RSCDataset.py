import os
import cv2
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset,DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import albumentations as A
from albumentations.pytorch import ToTensorV2

# from .transform import train_transform
# from .transform import val_transform
train_transform = A.Compose([
    # reszie
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        #A.HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
        A.CoarseDropout(p=0.2),
        A.Transpose(p=0.5)
    ]),
    A.Normalize(mean=(0.5, 0.5, 0.5,0.5), std=(0.5, 0.5, 0.5,0.5)),
    ToTensorV2(),
])
class RSCDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            img_trans = img_nd.transpose(2, 0, 1)
        except:
            print(img_nd.shape)
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')
        #print(mask_file)

        #image = cv2.imread(img_file[0], cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.imread(img_file[0], cv2.IMREAD_UNCHANGED)#四通道
        mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)

        #if self.transform:
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        return {
            'image': image,
            'label': mask.long()
        }

if __name__ == '__main__':
    data_dir = "../../yaogan/ctseg7/train/dataset"
    train_imgs_dir = os.path.join(data_dir, "train_images/")
    train_labels_dir = os.path.join(data_dir, "train_labels/")
    #
    train_data = RSCDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
    for batch_idx, batch_samples in enumerate(train_loader):
        image, target = batch_samples['image'], batch_samples['label']
        print(image.shape)
