
from re import A
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

def img_loader(img_path):
    img = Image.open(img_path)
    return img
class personData(Dataset):
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
        image_preds_all.append(image_preds)
    #
    image_preds_all=torch.cat(image_preds_all)
    return image_preds_all
if __name__=="__main__":
    df=pd.DataFrame(columns=['name',
        'upperLength','clothesStyles','hairStyles','lowerLength','lowerStyles','shoesStyles','towards',
        'upperBlack','upperBrown','upperBlue','upperGreen',
        'upperGray','upperOrange','upperPink','upperPurple','upperRed','upperWhite','upperYellow',
        'lowerBlack','lowerBrown','lowerBlue','lowerGreen','lowerGray','lowerOrange','lowerPink',
        'lowerPurple','lowerRed','lowerWhite','lowerYellow'])
    #
    color_columns=['upperBlack','upperBrown','upperBlue','upperGreen',
        'upperGray','upperOrange','upperPink','upperPurple','upperRed','upperWhite','upperYellow',
        'lowerBlack','lowerBrown','lowerBlue','lowerGreen','lowerGray','lowerOrange','lowerPink',
        'lowerPurple','lowerRed','lowerWhite','lowerYellow']
    #
    test_dir='../data/testB/'
    test_images = os.listdir(test_dir)
    #test_images.sort(key=lambda x: int(x[:-4]))
    test_paths=[os.path.join(test_dir,i) for i in test_images]
    preds_lst = []
    device = torch.device('cuda')
    save_dir='./'
    os.makedirs(save_dir,exist_ok=True)
    for tta in range(1):
        infer_ds = personData(test_paths,get_inference_transforms(224))
        infer_loader = torch.utils.data.DataLoader(
                infer_ds,
                batch_size=8,
                num_workers=4,
                shuffle=False,
                pin_memory=False,
            )
        for fold in [0]:
            print('Inference TTA:{} fold {} started'.format(tta,fold))
            model = ImgClassifier('convnext_large_in22ft1k', 46, pretrained=False).to(device)
            model.load_state_dict(torch.load('./ckpt/convnext_large_in22ft1k/convnext_large_in22ft1k_fold_0_last.pth'.format(fold)))
            preds_lst.append(inference(model, infer_loader, device))
            del model
            torch.cuda.empty_cache()
        for fold in [0]:
            print('Inference TTA:{} fold {} started'.format(tta,fold))
            model = ImgClassifier('convnext_xlarge_in22ft1k', 46, pretrained=False).to(device)
            model.load_state_dict(torch.load('./ckpt/convnext_xlarge_in22ft1k/convnext_xlarge_in22ft1k_fold_{}_last.pth'.format(fold)))
            preds_lst.append(inference(model, infer_loader, device))
            del model
            torch.cuda.empty_cache()
        for fold in [0]:
            print('Inference TTA:{} fold {} started'.format(tta,fold))
            model = ImgClassifier('swin_large_patch4_window7_224', 46, pretrained=False).to(device)
            model.load_state_dict(torch.load('./ckpt/swin_large_patch4_window7_224/swin_large_patch4_window7_224_fold_{}_last.pth'.format(fold)))
            preds_lst.append(inference(model, infer_loader, device))
            del model
            torch.cuda.empty_cache()
    #模型融合
    print(len(preds_lst))
    ratio=[0.2,0.4,0.4]
    assert abs(np.sum(ratio)-1)<1e-3
    for i in range(len(preds_lst)):
        if i==0:
            preds=ratio[i]*preds_lst[i]
        else:
            preds+=ratio[i]*preds_lst[i]
    #
    dic={'upperLength': {0: 'LongSleeve', 1: 'ShortSleeve', 2: 'NoSleeve'}, 
        'clothesStyles': {0: 'Solidcolor', 1: 'multicolour', 2: 'lattice'}, 
        'hairStyles': {0: 'Short', 1: 'Long', 2: 'middle', 3: 'Bald'}, 
        'lowerLength': {0: 'Skirt', 1: 'Trousers', 2: 'Shorts'}, 
        'lowerStyles': {0: 'Solidcolor', 1: 'lattice', 2: 'multicolour'}, 
        'shoesStyles': {0: 'Sandals', 1: 'Sneaker', 2: 'LeatherShoes', 3: 'else'}, 
        'towards': {0: 'right', 1: 'back', 2: 'front', 3: 'left'}}
    #
    upperLength_lst=[]
    clothesStyles_lst=[]
    hairStyles_lst=[]
    lowerLength_lst=[]
    lowerStyles_lst=[]
    shoesStyles_lst=[]
    towards_lst=[]

    color_lst1=[[np.nan]*len(test_paths) for _ in range(11)]
    color_lst2=[[np.nan]*len(test_paths) for _ in range(11)]
    name_lst=[]
    cnt=0
    print(preds.shape)
    for i in range(preds.shape[0]):
        output=preds[i]
        upperLength= torch.argmax(output[:3]).item()
        clothesStyles= torch.argmax(output[3:6]).item()
        hairStyles= torch.argmax(output[6:10]).item()
        lowerLength = torch.argmax(output[10:13]).item()
        lowerStyles = torch.argmax(output[13:16]).item()
        shoesStyles = torch.argmax(output[16:20]).item()
        towards = torch.argmax(output[20:24]).item()
        color1 = torch.argmax(output[24:35]).item()
        color2 = torch.argmax(output[35:]).item()


        upperLength_lst.append(dic['upperLength'][upperLength])
        clothesStyles_lst.append(dic['clothesStyles'][clothesStyles])
        hairStyles_lst.append(dic['hairStyles'][hairStyles])
        #
        lowerLength_lst.append(dic['lowerLength'][lowerLength])
        lowerStyles_lst.append(dic['lowerStyles'][lowerStyles])
        shoesStyles_lst.append(dic['shoesStyles'][shoesStyles])
        towards_lst.append(dic['towards'][towards])
        #
        color_lst1[color1][i]=1
        color_lst2[color2][i]=1
        name_lst.append(test_images[cnt])
        cnt+=1
    #
    df['name']=name_lst
    df['upperLength']=upperLength_lst
    df['clothesStyles']=clothesStyles_lst
    df['hairStyles']=hairStyles_lst
    df['lowerLength']=lowerLength_lst
    df['lowerStyles']= lowerStyles_lst
    df['shoesStyles']=shoesStyles_lst
    df['towards']=towards_lst
    #
    df['upperBlack']=color_lst1[0]
    df['upperBrown']=color_lst1[1]
    df['upperBlue']=color_lst1[2]
    df['upperGreen']=color_lst1[3]
    df['upperGray']=color_lst1[4]
    df['upperOrange']=color_lst1[5]
    df['upperPink']=color_lst1[6]
    df['upperPurple']=color_lst1[7]
    df['upperRed']=color_lst1[8]
    df['upperWhite']=color_lst1[9]
    df['upperYellow']=color_lst1[10]
    #
    df['lowerBlack']=color_lst2[0]
    df['lowerBrown']=color_lst2[1]
    df['lowerBlue']=color_lst2[2]
    df['lowerGreen']=color_lst2[3]
    df['lowerGray']=color_lst2[4]
    df['lowerOrange']=color_lst2[5]
    df['lowerPink']=color_lst2[6]
    df['lowerPurple']=color_lst2[7]
    df['lowerRed']=color_lst2[8]
    df['lowerWhite']=color_lst2[9]
    df['lowerYellow']=color_lst2[10]
    print(df)
    df.to_csv(save_dir+'/result.csv',index=False)