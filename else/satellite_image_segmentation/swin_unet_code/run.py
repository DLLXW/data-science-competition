# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
def get_infer_transform():
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5,0.5), std=(0.5, 0.5, 0.5,0.5)),
        ToTensorV2(),
    ])
    return transform
#
def inference(img_dir):
    transform=get_infer_transform()
    #image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    wd,ht,_=image.shape
    img = transform(image=image)['image']
    img=img.unsqueeze(0)
    #print(img.shape)
    with torch.no_grad():
        img=img.cuda()
        output = model(img)
    #
    pred = output.squeeze().cpu().data.numpy()
    pred = np.argmax(pred,axis=0)
    return pred,wd,ht
#
if __name__=="__main__":
    #input_dir,out_dir=sys.argv[1],sys.argv[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, default='/input_path')
    parser.add_argument('out_dir', type=str, default='/output_path')
    parser.add_argument('--num_classes', type=int,
                        default=47, help='output channel of network')                
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
    n_class=47
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    checkpoint_dir='cosine_epoch20.pth'
    print(checkpoint_dir)
    checkpoints=torch.load(checkpoint_dir)
    if 'state_dict' in checkpoints.keys():
        model.load_state_dict(checkpoints['state_dict'])
    else:
        model.load_state_dict(checkpoints)
    model.eval()
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    test_paths=glob.glob(input_dir+'/*')
    for per_path in tqdm(test_paths):
        result,wd,ht=inference(per_path)
        result=result.astype(np.uint8)
        # img=Image.fromarray(np.uint8(result))
        # img=img.convert('L')
        img=cv2.resize(result,(wd,ht),interpolation=cv2.INTER_NEAREST)
        out_path=os.path.join(out_dir,per_path.split('/')[-1][:-4]+'.png')
        cv2.imwrite(out_path,img)