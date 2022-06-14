#
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from net import ImgClassifier
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import csv

def img_loader(img_path):
    img = Image.open(img_path)
    return img
class carData(Dataset):
    def __init__(self, data_path,
                 transform=None):
        super(Dataset, self).__init__()
        self.samples = data_path
        self.transform=transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(image=np.array(img))['image']
        return img
def get_inference_transforms(input_size):
    return A.Compose([
            A.Resize(input_size,input_size),#276, 344
            #A.HorizontalFlip(p=0.5),
            #A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

@torch.no_grad()
def inference(model, data_loader, device):
    model.eval()
    image_preds_all = []
    img_path_all=[]
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (data) in pbar:
        imgs = data
        imgs=imgs.to(device).float()
        image_preds = model(imgs)
        #print(image_preds)
        image_preds_all.append(image_preds)
    #
    image_preds_all=torch.cat(image_preds_all)
    return image_preds_all
if __name__=="__main__":
    save_dir='./'
    os.makedirs(save_dir,exist_ok=True)
    df=pd.DataFrame(columns=['id','type','color','toward'])
    test_dir='../data/phase2B_test/'
    test_images = os.listdir(test_dir)
    test_images.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    #print(test_images)
    test_paths=[os.path.join(test_dir,i) for i in test_images]
    preds_lst = []
    device = torch.device('cuda')
    for tta in range(1):
        infer_ds = carData(test_paths,get_inference_transforms(224))
        infer_loader = torch.utils.data.DataLoader(
                infer_ds,
                batch_size=8,
                num_workers=4,
                shuffle=False,
                pin_memory=False,
            )
        # for fold in [0]:
        #     print('Inference TTA:{} fold {} started'.format(tta,fold))
        #     model = ImgClassifier('convnext_large_in22ft1k', 21, pretrained=False).to(device)
        #     model.load_state_dict(torch.load('./ckpt_reproduct/convnext_xlarge_in22ft1k_fold_{}_best.pth'.format(fold)))
        #     preds_lst.append(inference(model, infer_loader, device))
        #     del model
        #     torch.cuda.empty_cache()
        for fold in [0]:
            print('Inference TTA:{} fold {} started'.format(tta,fold))
            model = ImgClassifier('swin_large_patch4_window7_224', 21, pretrained=False).to(device)
            model.load_state_dict(torch.load('./ckpt/swin_large_patch4_window7_224/swin_large_patch4_window7_224_fold_{}_last.pth'.format(fold)))
            preds_lst.append(inference(model, infer_loader, device))
            del model
            torch.cuda.empty_cache()
        #
        for fold in [0]:
            print('Inference TTA:{} fold {} started'.format(tta,fold))
            model = ImgClassifier('convnext_xlarge_in22ft1k', 21, pretrained=False).to(device)
            model.load_state_dict(torch.load('./ckpt/convnext_xlarge_in22ft1k/convnext_xlarge_in22ft1k_fold_{}_last.pth'.format(fold)))
            preds_lst.append(inference(model, infer_loader, device))
            del model
            torch.cuda.empty_cache()
    #模型融合
    print(len(preds_lst))
    #ratio=[0.6/5]*5+[0.4/5]*5
    ratio=[0.4,0.6]
    assert abs(np.sum(ratio)-1)<1e-3
    for i in range(len(preds_lst)):
        if i==0:
            preds=ratio[i]*preds_lst[i]
        else:
            preds+=ratio[i]*preds_lst[i]
    #
    dic={'type': {0: 'van', 1: 'truck', 2: 'car', 3: 'suv', 4: 'coach', 5: 'bus', 6: 'engineering_car'}, 
    'color': {0: 'gray', 1: 'black', 2: 'indigo', 3: 'white', 4: 'red', 5: 'blue', 6: 'silvery', 7: 'brown', 8: 'gold', 9: 'yellow'},
    'toward': {0: 'left', 1: 'right', 2: 'back', 3: 'front'}}
    #
    type_lst=[]
    color_lst=[]
    toward_lst=[]
    name_lst=[]
    cnt=0
    for i in range(preds.shape[0]):
        output=preds[i]
        type= torch.argmax(output[:7]).item()
        color= torch.argmax(output[7:17]).item()
        toward= torch.argmax(output[17:]).item()
        type_lst.append(dic['type'][type])
        color_lst.append(dic['color'][color])
        toward_lst.append(dic['toward'][toward])
        #
        name_lst.append(test_images[cnt])
        cnt+=1
    #
    df['id']=name_lst
    df['type']=type_lst
    df['color']=color_lst
    df['toward']=toward_lst
    print(df)
    df.to_csv(save_dir+'/result.csv',index=False)