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
class sunDataset(Dataset):
    def __init__(self, data_dir,is_train=True):
        c_dir=data_dir+'/continuum'
        m_dir=data_dir+'/magnetogram'
        self.cls_dic={'alpha':0,'beta':1,'betax':2}
        self.c_paths=sorted(glob.glob(c_dir+'/*/*'))
        self.m_paths=sorted(glob.glob(m_dir+'/*/*'))
        #self.data_transforms = transforms.Compose([
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            #transforms.Normalize([0.48, 0.48, 0.48], [0.23, 0.23, 0.23])
        #])
        self.transform_train = A.Compose([
            A.Resize(height=224, width=224),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.RandomBrightness(limit=0.1, p=0.5),
            ], p=1),
            # A.GaussNoise(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(rotate_limit=1, p=0.5),
            # FancyPCA(alpha=0.1, p=0.5),
            # blur
            A.OneOf([
                A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3),
            ], p=0.5),
            # Pixels
            A.OneOf([
                A.IAAEmboss(p=0.5),
                A.IAASharpen(p=0.5),
            ], p=1),
            # Affine
            A.OneOf([
                A.ElasticTransform(p=0.5),
                A.IAAPiecewiseAffine(p=0.5),
            ], p=1),
            A.Normalize(mean=(0.81,0.51), std=(0.081,0.115), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])

        self.transform_valid = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=(0.81,0.51), std=(0.081,0.115), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        if is_train:
            self.data_transforms=self.transform_train
        else:
            self.data_transforms = self.transform_valid

    def __getitem__(self,index):
        #第index个样本
        sample_path1 = self.c_paths[index]
        sample_path2 = self.m_paths[index]
        #
        cls = sample_path1.split('/')[-2]
        label = self.cls_dic[cls]
        img1=cv2.imread(sample_path1,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(sample_path2, cv2.IMREAD_GRAYSCALE)
        #cv2.imwrite('yy1.jpg', img1)
        #cv2.imwrite('yy2.jpg', img2)
        #img1 = Image.open(sample_path1).convert('RGB')
        #img2 = Image.open(sample_path2).convert('RGB')
        img1=np.array(img1)
        img1=np.expand_dims(img1,axis=2)
        img2 = np.array(img2)
        img2 = np.expand_dims(img2, axis=2)
        img=np.concatenate((img1,img2),axis=2)
        img = self.data_transforms(image=img)['image']
        #img1=self.data_transforms(image=img1)['image']
        #img2=self.data_transforms(image=img2)['image']
        img = torch.cat((img, img[:1,:,:]), dim=0)
        return img,label

    def __len__(self):
        return len(self.c_paths)
class sunDatasetTest(Dataset):
    def __init__(self, data_dir,index):
        c_dir=data_dir+'/continuum'
        m_dir=data_dir+'/magnetogram'
        self.cls_dic={'alpha':0,'beta':1,'betax':2}
        self.c_paths=sorted(glob.glob(c_dir+'/*/*'))
        self.m_paths=sorted(glob.glob(m_dir+'/*/*'))
        if index == 0:
            self.trans = A.Compose([
                A.Resize(height=224, width=224),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
                ToTensorV2(p=1.0)
            ]),
        if index == 1:
            self.trans = A.Compose([
                A.Resize(height=224, width=224),
                A.RandomBrightness(limit=0.1, p=1),
                A.Normalize(mean=(0.81, 0.51), std=(0.081, 0.115), max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)])
        if index == 2:
            self.trans = A.Compose([
                A.Resize(height=224, width=224),
                A.HorizontalFlip(p=1),
                A.Normalize(mean=(0.81, 0.51), std=(0.081, 0.115), max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)])
        if index == 3:
            self.trans = A.Compose([
                A.Resize(height=224, width=224),
                A.RandomRotate90(p=1),
                A.Normalize(mean=(0.81, 0.51), std=(0.081, 0.115), max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)]
                )
        else:
            self.trans = A.Compose([
                A.Resize(height=224, width=224),
                A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3),
                A.Normalize(mean=(0.81, 0.51), std=(0.081, 0.115), max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)])

    def __getitem__(self,index):
        #第index个样本
        sample_path1 = self.c_paths[index]
        sample_path2 = self.m_paths[index]
        #
        cls = sample_path1.split('/')[3]
        label = self.cls_dic[cls]
        img1=cv2.imread(sample_path1,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(sample_path2, cv2.IMREAD_GRAYSCALE)
        img1=np.array(img1)
        img1=np.expand_dims(img1,axis=2)
        img2 = np.array(img2)
        img2 = np.expand_dims(img2, axis=2)
        img=np.concatenate((img1,img2),axis=2)
        img = self.trans(image=img)['image']
        img = torch.cat((img, img[:1, :, :]), dim=0)
        return img, label

    def __len__(self):
        return len(self.c_paths)
class sunDatasetInfer(Dataset):
    def __init__(self, data_dir):
        c_dir=data_dir+'/continuum'
        m_dir=data_dir+'/magnetogram'
        self.c_paths=sorted(glob.glob(c_dir+'/*'))
        self.m_paths=sorted(glob.glob(m_dir+'/*'))
        self.transform_valid = A.Compose([
            A.Resize(height=224, width=224),
            #A.RandomBrightness(limit=0.1, p=0.5),
            #A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.81,0.51), std=(0.081,0.115), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        self.data_transforms = self.transform_valid
    def __getitem__(self,index):
        #第index个样本
        sample_path1 = self.c_paths[index]
        sample_path2 = self.m_paths[index]
        #
        img1=cv2.imread(sample_path1,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(sample_path2, cv2.IMREAD_GRAYSCALE)
        img1=np.array(img1)
        img1=np.expand_dims(img1,axis=2)
        img2 = np.array(img2)
        img2 = np.expand_dims(img2, axis=2)
        img=np.concatenate((img1,img2),axis=2)
        img = self.data_transforms(image=img)['image']
        img = torch.cat((img, img[:1,:,:]), dim=0)
        return img,os.path.split(sample_path1)[1][:-5]

    def __len__(self):
        return len(self.c_paths)
'''
data_dir='dataset/val'
#sun=sunDataset(data_dir)
#img,label=sun.__getitem__(1)
#print(len(sun),img.shape)
image_datasets = sunDataset(data_dir)
dataset_loaders = torch.utils.data.DataLoader(image_datasets,
                                                      batch_size=1,
                                                      shuffle=True, num_workers=0)
for img,label in dataset_loaders:
    print(label)
    break
'''
# varify
'''
c_dir='trainset/continuum'
m_dir='trainset/magnetogram'
c_paths=sorted(glob.glob(c_dir+'/*/*'))
m_paths=sorted(glob.glob(m_dir+'/*/*'))
print(c_paths[1])

for i in range(len(c_paths)):
    cc=os.path.split(c_paths[i])[1]
    cc=cc.split('.')[:-2]
    mm = os.path.split(m_paths[i])[1]
    mm = mm.split('.')[:-2]
    if cc!=mm:
        print('xxxx')
#
img1=Image.open(c_paths[1]).convert('RGB')
img2=Image.open(m_paths[1]).convert('RGB')
img1 = np.array(img1)
img2 = np.array(img2)
img=img1[:,:,:2]
img[:,:,1]=img2[:,:,1]
#print(img[:,:,1]==img[:,:,0],img.shape)
'''