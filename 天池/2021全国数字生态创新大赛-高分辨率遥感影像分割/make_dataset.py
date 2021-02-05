import os
import shutil
from tqdm import tqdm
images_dir='/media/limzero/compete_datasets/images/'
mask_dir='/media/limzero/compete_datasets/masks_new'
ann_dir_train='./satellite_jpg/ann_dir/train/'
ann_dir_val='./satellite_jpg/ann_dir/val/'
ann_dir_train_val='./satellite_jpg/ann_dir/train_val/'
img_dir_train='./satellite_jpg/img_dir/train/'
img_dir_val='./satellite_jpg/img_dir/val/'
img_dir_train_val='./satellite_jpg/img_dir/train_val/'

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
    if i%val_interval==0:
        shutil.copy(os.path.join(images_dir,name),os.path.join(img_dir_val,name))
        shutil.copy(os.path.join(mask_dir,mask_name), os.path.join(ann_dir_val, mask_name))
        val_size+=1
    else:
        shutil.copy(os.path.join(images_dir, name), os.path.join(img_dir_train, name))
        shutil.copy(os.path.join(mask_dir,mask_name), os.path.join(ann_dir_train, mask_name))
        train_size+=1
    shutil.copy(os.path.join(images_dir, name), os.path.join(img_dir_train_val, name))
    shutil.copy(os.path.join(mask_dir, mask_name), os.path.join(ann_dir_train_val, mask_name))
print("train size:{},val size:{}".format(train_size,val_size))