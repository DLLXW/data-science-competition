

import cv2
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2

def get_train_transforms(height_size,width_size):
    return A.Compose([
            A.Resize(height_size,width_size),#[height,width]
            #A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.25),
            #A.ShiftScaleRotate(rotate_limit=10,p=0.5),
            #A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.OneOf([
                    # 模糊相关操作
                    A.MotionBlur(p=.75),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.Blur(blur_limit=3, p=0.75),
                ], p=0.25),
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
                    ],p=0.25),
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

class RubbishDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 in_channels=3,
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.in_channels=in_channels#输入通道数量
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['target'].values
            #print(self.labels)
            if one_hot_label is True:
                self.labels = np.eye(self.df['target'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
            target=[float(i) for i in target.strip('[]').split(',')]
        #print("{}/{}".format(self.data_root, self.df.loc[index]['file_name']),target_str)
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['name']))
        #print(img.shape)
        if self.transforms:
            img = self.transforms(image=img)['image']
        if self.output_label == True:
            return img,torch.from_numpy(np.array(target))
        else:
            return img

if __name__=="__main__":
    data_root='/home/trojanjet/baidu_qyl/data_lake/person/input/train/train1A/'
    train_csv = pd.read_csv('../train_df.csv')
    train_ = train_csv
    train_ds = RubbishDataset(train_, data_root, transforms=get_train_transforms(384), output_label=True,input_size=384, )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=4,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
        #sampler=BalanceClassSampler(labels=train_['class_id'].values, mode="downsampling")
    )
    for step,(imgs, image_labels) in enumerate(train_loader):
        #[c,h,w]->[h,w,c]
        img=imgs[0].permute(1,2,0)
        print(img.shape,image_labels[3])
        cv2.imwrite('aug.jpg',img.numpy())
        break