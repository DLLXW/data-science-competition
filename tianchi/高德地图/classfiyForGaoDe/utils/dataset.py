import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
import cv2
import numpy as np
from torchvision import datasets, models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import FancyPCA
class roadDataset(Dataset):
    def __init__(self, data_dir,is_train=True,image_size=224):
        self.paths=sorted(glob.glob(data_dir+'/*/*'))
        self.transform_train = A.Compose([
            A.Resize(height=image_size, width=image_size),
            #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.446, 0.469, 0.472), std=(0.326, 0.330, 0.338), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])

        self.transform_valid = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.446, 0.469, 0.472), std=(0.326, 0.330, 0.338), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        if is_train:
            self.data_transforms=self.transform_train
        else:
            self.data_transforms = self.transform_valid

    def __getitem__(self,index):
        #
        sample_path = self.paths[index]
        #
        cls = sample_path.split('/')[-2]

        label = int(cls)

        img=Image.open(sample_path)
        img = img.convert('RGB')
        img=np.array(img)

        img = self.data_transforms(image=img)['image']

        return img,label#,sample_path

    def __len__(self):
        return len(self.paths)
class roadDatasetInfer(Dataset):
    def __init__(self, data_dir,image_size=224):
        self.paths=sorted(glob.glob(data_dir+'/*/*'))
        self.data_transforms = A.Compose([
            A.Resize(height=image_size, width=image_size),
            #A.HorizontalFlip(p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.Normalize(mean=(0.446, 0.469, 0.472), std=(0.326, 0.330, 0.338), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])

    def __getitem__(self,index):
        #
        sample_path = self.paths[index]
        img=Image.open(sample_path)
        img = img.convert('RGB')
        img=np.array(img)
        img = self.data_transforms(image=img)['image']
        return img,sample_path

    def __len__(self):
        return len(self.paths)

'''
data_dir='test_jpg_input/'
sun=sunDatasetInfer(data_dir)
#img,label=sun.__getitem__(1)
#print(len(sun),img.shape)
#image_datasets = sunDataset(data_dir)
dataset_loaders = torch.utils.data.DataLoader(sun,batch_size=1,shuffle=False, num_workers=0)
for img,path in dataset_loaders:
    print(path)
    print(img.shape)
    break
'''
