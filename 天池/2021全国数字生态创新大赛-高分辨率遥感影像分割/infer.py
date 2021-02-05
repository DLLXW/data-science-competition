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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import colorEncode
import torch.nn as nn
from torch.cuda.amp import autocast
def visualize_result(img_dir, pred):
    #
    img=cv2.imread(img_dir)
    colors = loadmat('demo/color150.mat')['colors']
    names = {
            1: "耕地",
            2: "林地",
            3: "草地",
            4: "道路",
            5: "城镇建设用地",
            6: "农村建设用地",
            7: "工业用地",
            8: "构筑物",
            9: "水域",
            10: "裸地"
        }
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    #
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    #print(pred_color.shape)
    #pred_color=cv2.resize(pred_color,(256,256))
    im_vis = np.concatenate((img, pred_color), axis=1)

    #
    #img_name=image_demo_dir.split('/')[-1]
    save_dir,name=os.path.split(img_dir)
    Image.fromarray(im_vis).save('demo/256x256_deeplab_44.png')
def get_infer_transform():
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return transform
#
def inference(img_dir):
    transform=get_infer_transform()
    image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(image=image)['image']
    img=img.unsqueeze(0)
    #print(img.shape)
    with torch.no_grad():
        img=img.cuda()
        output = model(img)
    #
    pred = output.squeeze().cpu().data.numpy()
    pred = np.argmax(pred,axis=0)
    return pred+1

class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()
        self.model = smp.UnetPlusPlus(
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
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
    n_class=10
    model=seg_qyl(model_name,n_class).cuda()
    model= torch.nn.DataParallel(model)
    checkpoints=torch.load('outputs/efficientnet-b6/ckpt/checkpoint-epoch44.pth')
    model.load_state_dict(checkpoints['state_dict'])
    model.eval()
    use_demo=False
    assert_list=[1,2,3,4,5,6,7,8,9,10]
    if use_demo:
        img_dir='demo/000097.jpg'
        pred=inference(img_dir)
        infer_start_time = time.time()
        visualize_result(img_dir, pred)
        #
    else:
        out_dir='result/results/'
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        test_paths=glob.glob('/media/limzero/compete_datasets/test_jpg/*')
        for per_path in tqdm(test_paths):
            result=inference(per_path)
            img=Image.fromarray(np.uint8(result))
            img=img.convert('L')
            #print(out_path)
            out_path=os.path.join(out_dir,per_path.split('/')[-1][:-4]+'.png')
            img.save(out_path)