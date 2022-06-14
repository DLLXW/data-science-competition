
import cv2
from PIL import Image
import torchvision
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms as T


def get_train_transforms(input_size):
    return T.Compose([
            T.Resize((input_size,input_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
def get_valid_transforms(input_size):
   return T.Compose([
        T.Resize((input_size,input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
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
        sample_path  = "{}/{}".format(self.data_root, self.df.loc[index]['file_name'])
        img = Image.open(sample_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        if self.output_label == True:
            return img,target
        else:
            return img

if __name__=="__main__":
    data_root='/media/limzero/ssd512/qyl/competition_code/sandong/data/data/jishui_data/train_jpgs'
    train_csv = pd.read_csv('./train_df.csv')
    train_ = train_csv
    train_ds = RubbishDataset(train_, data_root, transforms=get_train_transforms(224), output_label=True,input_size=224, )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    for i, (data, label) in enumerate(train_loader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        #cv2.imshow('img', img)
        img *= np.array([0.5, 0.5, 0.5])*255
        img += np.array([0.5, 0.5, 0.5])*255
        #img += np.array([1, 1, 1])
        #img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imwrite('img.jpg', img)
        break