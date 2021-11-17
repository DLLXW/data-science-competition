import os
import shutil
import cv2
from PIL import Image
import numpy as np
def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:#1440<2560
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
def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis
if __name__=="__main__":
    min_size=512
    max_size=2048
    handcraft_distance=True#手工标注的dis
    train=open('../train.txt','w')
    val=open('../val.txt','w')
    root_dir="../../data/140/images/"
    root_dir_ann="../../data/140/point_npy/"
    images=os.listdir(root_dir)
    train_data="../../data/140/train/"
    val_data="../../data/140/val/"
    if not os.path.exists(train_data):os.makedirs(train_data)
    if not os.path.exists(val_data):os.makedirs(val_data)
    for i in range(len(images)):
        name=images[i]
        cur_path=os.path.join(root_dir,name)
        im = Image.open(cur_path)
        im_w, im_h = im.size
        im = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
        points=np.load(os.path.join(root_dir_ann,name.replace('jpg','npy')))
        if rr != 1.0:
            im = cv2.resize(im, (im_w, im_h), cv2.INTER_CUBIC)
            #print(im.shape)
            points = points * rr
        #
        if i%10==0:
            val.write(name+'\n')
            cv2.imwrite(os.path.join(val_data,name),im)
            np.save(os.path.join(val_data,name.replace('jpg','npy')),points)
        else:
            train.write(name+'\n')
            if not handcraft_distance:
                try:
                    dis = find_dis(points)
                    points = np.concatenate((points, dis), axis=1)
                    cv2.imwrite(os.path.join(train_data,name),im)
                    np.save(os.path.join(train_data,name.replace('jpg','npy')),points)
                except:
                    dis=np.array([[100]]*points.shape[0])
                    print(dis.shape)
                    points = np.concatenate((points, dis), axis=1)
                    cv2.imwrite(os.path.join(train_data,name),im)
                    np.save(os.path.join(train_data,name.replace('jpg','npy')),points)
            else:
                cv2.imwrite(os.path.join(train_data,name),im)
                np.save(os.path.join(train_data,name.replace('jpg','npy')),points)
    #
    train.close()
    val.close()