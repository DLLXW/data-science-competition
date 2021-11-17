#
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis
def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

if __name__ == '__main__':
    #
    min_size = 512
    max_size = 2048
    train_val_root="../../data/100/"#train/"
    train_data=glob(train_root+"/*")
    train_data_save_dir="../../data/100/train/"
    for mode in ["train","val"]
    for path in train_data:
        if not path.endswith('npy'):
            #
            im = Image.open(path)
            im_w, im_h = im.size
            im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
            im = np.array(im)
            if rr != 1.0:
                im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
                print(im.shape)
                points = points * rr
                cv2.imwrite(path,im)
        else:
            points=np.load(path)
            dis = find_dis(points)
            points = np.concatenate((points, dis), axis=1)
            np.save(path,points)