'''
2020/6/15,标注文件转换xml转txt（vol to yolo）转完后需添加labels文件，即数字序号对应的标签名。

'''

import json
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np
import shutil
classes = ['chicken',]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    # x = (box[0] + box[1])/2.0 - 1
    # y = (box[2] + box[3])/2.0 - 1
    # w = box[1] - box[0]
    # h = box[3] - box[2]
    x=box[0]+box[2]/2
    y=box[1]+box[3]/2
    w=box[2]
    h=box[3]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    if w>=1:
        w=0.99
    if h>=1:
        h=0.99
    return (x,y,w,h)

def convert_annotation(name,mode='train'):
    cnt=0
    path=os.path.join(root_npy_dir,name)
    anns =np.load(path)
    xywh=anns[:,:4]
    width = int(anns[0][4])
    height = int(anns[0][5])
    txtname=name.replace('npy','txt')
    print('ann length:',len(anns))
    if mode=='train':
        txtfile = os.path.join(txtpath_train,txtname)
        shutil.copy(os.path.join(root_image_dir,name.replace('npy','jpg')),os.path.join(imgpath_train,name.replace('npy','jpg')))
    else:
        txtfile = os.path.join(txtpath_val,txtname)
        shutil.copy(os.path.join(root_image_dir,name.replace('npy','jpg')),os.path.join(imgpath_val,name.replace('npy','jpg')))
    with open(txtfile, "w+" ,encoding='UTF-8') as out_file:
        out_file.truncate()
        for obj in xywh:
            cls_id = 0
            b = (float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]))
            bb = convert((width,height), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            cnt+=1
    print('cnt / ann length:',len(anns),cnt)
def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
if __name__ == "__main__":
    root_image_dir='../../../data/90/images/'#'../../rare_sample/'
    root_npy_dir ='../../../data/90/bbox_npy/' #已知的npy的标注
    txtpath_train = '../datasets/coco90/labels/train2017'
    imgpath_train = '../datasets/coco90/images/train2017'
    txtpath_val = '../datasets/coco90/labels/val2017'
    imgpath_val = '../datasets/coco90/images/val2017'
    make_dir(txtpath_train)
    make_dir(imgpath_train)
    make_dir(txtpath_val)
    make_dir(imgpath_val)
    list_names=os.listdir(root_npy_dir)
    print(len(list_names))
    for i in range(0,len(list_names)) :
        name=list_names[i]
        if i%10==0:
            convert_annotation(name,'val')
        else:
            convert_annotation(name,'train')