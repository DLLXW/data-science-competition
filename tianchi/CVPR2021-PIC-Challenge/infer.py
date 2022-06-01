
import os
import cv2
import numpy as np 
import torch
from config import Config
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from networks import landmarkRegressNet,landmarkPfld
class inferDataset(torch.utils.data.Dataset):

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
        if self.transform:
            augmented = self.transform(image=image_left)
            image_left = augmented['image']
            augmented = self.transform(image=image_mid)
            image_mid = augmented['image']
            augmented = self.transform(image=image_right)
            image_right = augmented['image']
        path='/'.join(left_mid_right[0].split('/')[-5:-3])
        return {
            'image' : [image_left,image_mid,image_right],
            'path':path
        }
if __name__=="__main__":
    model_name = 'tf_efficientnet_b2'#efficientnet-b2
    if model_name=='pfld':
        model =landmarkPfld()
    else:
        model  = landmarkRegressNet(
                 model_arch=model_name,
                 landmarks=318,
                 pretrained=True)
    #
    checkpoints=torch.load('ckpt/tf_efficientnet_b2/tf_efficientnet_b2_60.pth')
    model.load_state_dict(checkpoints)
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    inferset = inferDataset( root='/home/limzero/qyl/3Dface/infer_crop/',
                            list_dir='dataset/infer_crop.txt',
                            input_size=224,
                            transform = None
                            )

    inferloader = torch.utils.data.DataLoader(
                inferset,
                batch_size = 1,
                num_workers = 0,
                shuffle = False,
        )
    save_dir='submit/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_txt=open(save_dir+'/sub_0121_60epoch.txt','w')
    for i, data in enumerate(inferloader):
        image_left,image_mid,image_right=data['image']
        image_left=image_left.cuda()
        image_mid=image_mid.cuda()
        image_right=image_right.cuda()
        output=model(image_left,image_mid,image_right)
        pre=output[0].detach().cpu().numpy()
        path=data['path'][0]
        print(path)
        result_txt.write(path+'\n')
        for i in range(106):
            result_txt.write(str(pre[i*3])+' '+str(pre[i*3+1])+' '+str(pre[i*3+2])+'\n')
        #break