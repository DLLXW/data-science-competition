# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
#from skimage import io
from utils import colorEncode
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
def visualize_result(img_dir, pred):
    #
    img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/1023
    colors = loadmat('demo/color150.mat')['colors']
    names = {
            1: "耕地",
            2: "林地",
            3: "草地",
            4: "水域",
            5: "城乡-工矿-居民用地",
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
    pred_color=cv2.resize(pred_color,(img.shape[0],img.shape[1]))
    #im_vis = np.concatenate((img, pred_color), axis=1)
    #
    return pred_color

def get_infer_transform():
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1023.0, p=1.0),
        ToTensorV2(p=1.0),
    ],p=1.)
    return transform
#
def inference(img_dir):
    transform=get_infer_transform()
    #image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    pred = np.argmax(pred,axis=0)+1
    return pred,wd,ht
if __name__=="__main__":
    #input_dir,out_dir=sys.argv[1],sys.argv[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../data/test_roundA/')
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--num_classes', type=int,
                        default=6, help='output channel of network')                
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', 
        metavar="FILE", help='path to config file', )
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
    args = parser.parse_args()
    config = get_config(args)
    input_dir=args.input_dir
    out_dir=args.out_dir
    print(input_dir,out_dir)
    n_class=6
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model= torch.nn.DataParallel(model)
    checkpoint_dir='./outputs/swin_tiny/ckpt/cosine_epoch92.pth'
    print(checkpoint_dir)
    checkpoints=torch.load(checkpoint_dir)
    if 'state_dict' in checkpoints.keys():
        model.load_state_dict(checkpoints['state_dict'])
    else:
        model.load_state_dict(checkpoints)
    model.eval()
    use_demo=False
    if use_demo:
        img_dir='./demo/000006_GF.tif'
        img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/1023
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

    else:
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        test_paths=glob.glob(input_dir+'/*')
        for per_path in tqdm(test_paths):
            result,wd,ht=inference(per_path)
            result=result.astype(np.uint8)
            # img=Image.fromarray(np.uint8(result))
            # img=img.convert('L')
            img=cv2.resize(result,(wd,ht),interpolation=cv2.INTER_NEAREST)
            out_path=os.path.join(out_dir,per_path.split('/')[-1].replace('GF','LT'))
            cv2.imwrite(out_path,img)