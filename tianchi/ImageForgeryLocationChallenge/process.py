import os
import shutil
from tqdm import tqdm
import cv2
import collections
import numpy as np
images_dir='../data/train/img'
mask_dir='../data/train/mask'
ann_dir_train='../data/train_val_split/train_labels/'
ann_dir_val='../data/train_val_split/val_labels/'
ann_dir_train_val='../data/train_val_split/train_val_labels/'
img_dir_train='../data/train_val_split/train_images/'
img_dir_val='../data/train_val_split/val_images/'
img_dir_train_val='../data/train_val_split/train_val_images/'

if not os.path.exists(ann_dir_train):os.makedirs(ann_dir_train)
if not os.path.exists(ann_dir_val):os.makedirs(ann_dir_val)
if not os.path.exists(ann_dir_train_val):os.makedirs(ann_dir_train_val)
if not os.path.exists(img_dir_train):os.makedirs(img_dir_train)
if not os.path.exists(img_dir_val):os.makedirs(img_dir_val)
if not os.path.exists(img_dir_train_val):os.makedirs(img_dir_train_val)
val_ratio=0.2
val_interval=int((1/val_ratio))
train_size=0
val_size=0
names=os.listdir(images_dir)
for i in tqdm(range(len(names))):
    name=names[i]
    mask_name=name[:-4]+'.png'
    mask=cv2.imread(os.path.join(mask_dir,mask_name), cv2.IMREAD_GRAYSCALE)
    mask[mask<=127]=1
    mask[mask>127]=2
    # tmp=np.array(mask).reshape(mask.shape[0]*mask.shape[1],).tolist()
    # tmp=dict(collections.Counter(tmp))
    # tmp=sorted(tmp.items(), key=lambda x: x[1], reverse=True)
    # print(tmp)
    if i%val_interval==0:
        shutil.copy(os.path.join(images_dir,name),os.path.join(img_dir_val,name))
        cv2.imwrite(os.path.join(ann_dir_val, mask_name),mask)
        val_size+=1
    else:
        shutil.copy(os.path.join(images_dir, name), os.path.join(img_dir_train, name))
        cv2.imwrite(os.path.join(ann_dir_train,mask_name),mask)
        train_size+=1
    shutil.copy(os.path.join(images_dir, name), os.path.join(img_dir_train_val, name))
    cv2.imwrite(os.path.join(ann_dir_train_val, mask_name), mask)
    #break
print("train size:{},val size:{}".format(train_size,val_size))