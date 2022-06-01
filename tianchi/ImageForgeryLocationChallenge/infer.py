# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import time
from io import BytesIO
import base64
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
from tqdm import tqdm
import glob
import os
from scipy.io import loadmat
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import colorEncode
import torch.nn as nn
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
def visualize_result(img_dir, pred):
    #
    img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/1023
    colors = loadmat('demo/color150.mat')['colors']
    names = {
            0: "背景",
            1: "篡改",
        }
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    #
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx]]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.001:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint16)

    # aggregate images and save
    #print(pred_color.shape)
    pred_color=cv2.resize(pred_color,(img.shape[1],img.shape[0]))
    #im_vis = np.concatenate((img, pred_color), axis=1)
    #
    return pred_color
def get_infer_transform():
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    return transform
#
def inference(img_dir):
    transform=get_infer_transform()
    image = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ht,wd,_=image.shape
    img = transform(image=image)['image']
    img=img.unsqueeze(0)
    #print(img.shape)
    with torch.no_grad():
        img=img.cuda()
        out1 = model(img)
        out2 = model(torch.flip(img, dims=[2]))
        out2 = torch.flip(out2, dims=[2])
        out3 = model(torch.flip(img, dims=[3]))
        out3 = torch.flip(out3, dims=[3])
        out = (out1 + out2 + out3) / 3.0 
    #
    pred = out.squeeze().cpu().data.numpy()
    pred = np.argmax(pred,axis=0)
    return pred,wd,ht

class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()
        self.model = smp.UnetPlusPlus(
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    #
    @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x
#
if __name__=="__main__":
    model_name = 'efficientnet-b6'#efficientnet-b4
    n_class=2
    model=seg_qyl(model_name,n_class).cuda()
    model= torch.nn.DataParallel(model)
    checkpoint_dir='./outputs/efficientnet-b6/ckpt/checkpoint-best.pth'
    print(checkpoint_dir)
    checkpoints=torch.load(checkpoint_dir)
    if 'state_dict' in checkpoints.keys():
        model.load_state_dict(checkpoints['state_dict'])
    else:
        model.load_state_dict(checkpoints)
    model.eval()
    use_demo=False
    #assert_list=[1,2,3,4,5,6,7,8,9,10]
    if use_demo:
        img_dir='./demo/4.jpg'
        img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('xx.png',img)
        plt.figure(figsize=(18,16))
        plt.subplot(121)
        plt.imshow(img)
        pred,wd,ht=inference(img_dir)
        mask=visualize_result(img_dir,pred)
        plt.subplot(122)
        plt.imshow(mask)
        save_dir='demo/'+img_dir.split('/')[-1]+'_vis.png'
        plt.savefig(save_dir,dpi=300)
        #
    else:
        out_dir='images/images/'
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        test_paths=glob.glob('../data/test/img/*')
        for per_path in tqdm(test_paths):
            result,wd,ht=inference(per_path)
            result=result.astype(np.uint8)
            result[result==1]=255
            # img=Image.fromarray(np.uint8(result))
            # img=img.convert('L')
            img=cv2.resize(result,(wd,ht),interpolation=cv2.INTER_NEAREST)
            out_path=os.path.join(out_dir,per_path.split('/')[-1].replace('jpg','png'))
            cv2.imwrite(out_path,img)