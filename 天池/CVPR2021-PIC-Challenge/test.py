#!/usr/bin/env python3
# coding: utf-8

import os
import torch
import torchvision.transforms as transforms
from networks.resnet import resnet50, resnet34,  resnet18
#from face_data_loaders import ResizeGjz, ToTensorGjz, NormalizeGjz
import numpy as np
import cv2
import scipy.io as sio
import argparse
import torch.backends.cudnn as cudnn
import pdb
import os.path as osp
import math
import numpy.linalg as LA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.io import imread

def readImageList(imageListFile):
    imageList = []
    with open(imageListFile,'r') as fi:
        while(True):
            line = fi.readline().strip()
            if not line:
                break
            imageList.append(line)
    print( 'read imageList done image num ', len(imageList))
    return imageList

def estimate_pose(points_static, points_to_transform):
    #pdb.set_trace()
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3,1)
    t1 = -np.mean(p1, axis=1).reshape(3,1)
    

    p0c = p0+t0
    p1c = p1+t1

    covariance_matrix = p0c.dot(p1c.T)
    U,S,V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:,2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0)**2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0)**2))

    s = (rms_d0/rms_d1)
    P = s*np.eye(3).dot(R)
    t_final = P.dot(t1) -t0
    P = np.c_[P, t_final]

    return P

def main(args):
    pts_num = args.num_points
    crop_dim = args.img_size
    root_path = args.root_path
    DST_PTS = np.float32([[0,0], [0,crop_dim - 1], [crop_dim - 1, 0]]) 
    # 1. load pre-tained model
    checkpoint_fp = args.checkpoint 

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = eval(args.net)(num_classes=args.num_points*3)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        device = torch.device("cuda:%d"%(args.devices_id))
        cudnn.benchmark = True
        #model = model.cuda()
        model = model.to(device)
    model.eval()

    # 3. forward
    transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])


    lines = []
    with open(args.test_list,'r') as fi:
        while(True):
            line = fi.readline().strip()
            if not line:
                break
            lines.append(line)

    num = len(lines)//6

    fop = open('evaluation/val_baseline.txt', 'w')

    ind = 0
    for nn in range(num):
        point3d = []
        for k in range(3):
            img_path = os.path.join(root_path, lines[ind])
            ind = ind + 1
            print('%s\n'%img_path)
            img_src = cv2.imread(img_path)
            [img_h,img_w,img_c] = img_src.shape

            roi = [int(x) for x in lines[ind].split()]
            ind = ind + 1

            w = int(roi[2] - roi[0] + 1)
            h = int(roi[3] - roi[1] + 1)
            src_pts = np.float32([[roi[0], roi[3]], [roi[2], roi[3]], [roi[0], roi[1]]])
            dst_pts = np.float32([[0, 0], [0, crop_dim], [crop_dim, 0]])
            tform = cv2.getAffineTransform(src_pts, dst_pts)
            #pdb.set_trace()
            im = cv2.warpAffine(img_src,tform,(crop_dim, crop_dim)) 

            input = transform(im).unsqueeze(0)

            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.to(device)
                param = model(input)
                pre_points = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            pre_points = np.array(pre_points)
            pre_points = np.reshape(pre_points, [-1, 3])

            point3d.append(pre_points)

            ##save pre result 
            
        s_str = img_path.split('/')
        fold_name = os.path.join(*s_str[-5:-3])
        fop.write('%s\n'%fold_name)
        point3d_mean = np.mean(point3d, 0)
        for pts in point3d_mean:
            fop.write('%f %f %f\n'%(pts[0], pts[1], pts[2]))

    fop.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D inference pipeline')
    parser.add_argument('--root-path', default='you dataset path',
                        help='root image fold path', type=str)
    parser.add_argument('--test-list', default='data/val_list.txt',
                        help='image list path', type=str)
    parser.add_argument('--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--devices-id', default=0, type=int)
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--checkpoint', default='models/baseline.pth.tar', type=str)
    parser.add_argument('--num-points', default=106, type=int)
    args = parser.parse_args()
    main(args)

