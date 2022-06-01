
import os
import cv2
import numpy as np 
import torch
from config import Config
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(input_size=512):
    return A.Compose([
         A.Resize(input_size, input_size, always_apply=True),
         A.HorizontalFlip(p=0.5),
         A.RandomBrightness(limit=(0.09, 0.6), p=0.5),
         A.Normalize(
             mean = [0.485, 0.456, 0.406],
             std = [0.229, 0.224, 0.225]
         ),
         ToTensorV2(p=1.0)
     ])
def get_val_transforms(input_size=512):
    return albumentations.Compose([
         A.Resize(input_size, input_size, always_apply=True),
         A.Normalize(
             mean = [0.485, 0.456, 0.406],
             std = [0.229, 0.224, 0.225]
         ),
         ToTensorV2(p=1.0)
     ])
class faceDataset(torch.utils.data.Dataset):

    def __init__(self, root,list_dir, input_size,transform=None):
        self.root_dir = root
        with open(os.path.join(list_dir), 'r') as fd:
            self.imgPaths = [line.strip('\n') for line in fd.readlines()]
        #
        self.transform = A.Compose([
                A.Resize(input_size, input_size, always_apply=True),
                A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
                ),
         ToTensorV2(p=1.0)
     ])
        #print(self.imgPaths)
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        left_mid_right = [os.path.join(self.root_dir,i) for i in self.imgPaths[idx].split(' ')]
        image_left = cv2.imread(left_mid_right[0])
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
        image_mid = cv2.imread(left_mid_right[1])
        image_mid = cv2.cvtColor(image_mid, cv2.COLOR_BGR2RGB)
        image_right = cv2.imread(left_mid_right[2])
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
        #
        label=[]
        with open('/'.join(left_mid_right[0].split('/')[:-2])+'/fusion/nicp_106_pts.txt','r') as f:
            lines=f.readlines()
        for line  in lines:
            line=line.strip('\n')
            line=[float(i) for i in line.split(' ')]
            label+=line
        label=np.array(label)

        if self.transform:
            augmented = self.transform(image=image_left)
            image_left = augmented['image']
            augmented = self.transform(image=image_mid)
            image_mid = augmented['image']
            augmented = self.transform(image=image_right)
            image_right = augmented['image']

        return {
            'image' : [image_left,image_mid,image_right],
            'label' : torch.from_numpy(label)
        }
if __name__=="__main__":
    opt=Config()
    trainset = faceDataset( root=opt.root,
                            list_dir=opt.val_list_dir,
                            input_size=opt.input_size,
                            transform = None
                            )

    trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size = 1,
                num_workers = 0,
                shuffle = True,
        )
    for i, data in enumerate(trainloader):
        print(data['label'].shape)
        print(data['image'][0].shape)
        break