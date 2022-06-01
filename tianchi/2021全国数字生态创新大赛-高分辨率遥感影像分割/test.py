# -*- coding: utf-8 -*-
from model.efficientunet import *
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
from dataset import RSCDataset
from dataset import val_transform
from torch.utils.data import Dataset, DataLoader
from utils import IOUMetric
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_model(model_path):
    model = get_efficientunet_b3(out_channels=10, concat_input=True, pretrained=False)
    #model_1 = get_efficientunet_b4(out_channels=10, concat_input=True, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path)
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    #model_1 = model_1.to(device)
    #model_1.load_state_dict(checkpoint['b4_state_dict']['state_dict'])

    model.eval()
    return model
#
def cal_metrics(pred_label, gt):

    def _generate_matrix(gt_image, pre_image, num_class=10):
        mask = (gt_image >= 0) & (gt_image < num_class)#ground truth中所有正确(值在[0, classe_num])的像素label的mask
        #print(mask)
        label = num_class * gt_image[mask].astype('int') + pre_image[mask] 
        #print(label.shape)
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class, num_class)#21 * 21(for pascal)
        #print(confusion_matrix.shape)
        return confusion_matrix

    def _Class_IOU(confusion_matrix):
        MIoU = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix))
        return MIoU

    confusion_matrix = _generate_matrix(gt.astype(np.int8), pred_label.astype(np.int8))
    miou = _Class_IOU(confusion_matrix)
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return miou, acc
#
if __name__=="__main__":
    iou=IOUMetric(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path='/media/limzero/qyl/HWCC2020_RS_segmentation/outputs/efficient-b3/ckpt/checkpoint-epoch2.pth'
    model=load_model(model_path)
    data_dir = "/media/limzero/qyl/mmsegmentation/data/satellite_jpg/"
    val_imgs_dir = os.path.join(data_dir, "img_dir/val/")
    val_labels_dir = os.path.join(data_dir, "ann_dir/val/")
    valid_data = RSCDataset(val_imgs_dir, val_labels_dir, transform=val_transform)
    valid_loader = DataLoader(dataset=valid_data, batch_size=16, shuffle=False, num_workers=1)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_samples in tqdm(enumerate(valid_loader)):
            data, target = batch_samples['image'], batch_samples['label']
            print(data.shape,target.shape)
            data= Variable(data.to(device))
            pred = model(data)
            pred=pred.cpu().data.numpy()
            target=target.numpy()
            pred= np.argmax(pred,axis=1)
            iou.add_batch(pred,target)
            break
    #
    acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
    print(iu)
    print(mean_iu)