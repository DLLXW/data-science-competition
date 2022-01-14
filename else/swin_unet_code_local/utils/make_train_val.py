import cv2
import os
import random
from tqdm import tqdm
import shutil
import numpy as np
import collections

image_dir='../../data/初赛训练_GF/'
mask_dir='../../data/初赛训练_LT/'
save_img_train_dir='../../data/train_images/'
save_img_val_dir='../../data/val_images/'
save_label_train_dir='../../data/train_labels/'
save_label_val_dir='../../data/val_labels/'
os.makedirs(save_img_train_dir,exist_ok=True)
os.makedirs(save_img_val_dir,exist_ok=True)
os.makedirs(save_label_train_dir,exist_ok=True)
os.makedirs(save_label_val_dir,exist_ok=True)

def EDA():
    names=os.listdir(mask_dir)
    random.shuffle(names)
    class_dict=[0]*6
    for i in tqdm(range(len(names))):
        name=names[i]
        mask=cv2.imread(os.path.join(mask_dir,name),cv2.IMREAD_UNCHANGED)
        mask=np.array(mask).reshape(mask.shape[0]*mask.shape[1],).tolist()
        tmp=dict(collections.Counter(mask))
        for key in tmp.keys():
            class_dict[key-1]+=tmp[key]
    class_ratio=np.array(class_dict)/np.sum(class_dict)
    print(class_ratio)#[0.5953849  0.10441404 0.00519028 0.09199187 0.19774062 0.00527829]

if __name__=="__main__":
    #查看各个类别比例
    #EDA()
    #切分验证集/训练集
    train_size=0
    val_size=0
    names=os.listdir(image_dir)
    random.shuffle(names)
    #print(names)
    for i in tqdm(range(len(names))):
        name=names[i]
        mask_name=name.replace('GF','LT')
        if i%10==0:
            shutil.copy(os.path.join(image_dir,name),os.path.join(save_img_val_dir,name))
            shutil.copy(os.path.join(mask_dir,mask_name), os.path.join(save_label_val_dir, name))
            val_size+=1
        else:
            shutil.copy(os.path.join(image_dir, name), os.path.join(save_img_train_dir, name))
            shutil.copy(os.path.join(mask_dir,mask_name), os.path.join(save_label_train_dir, name))
            train_size+=1
        #
    print("train size:{},val size:{}".format(train_size,val_size))