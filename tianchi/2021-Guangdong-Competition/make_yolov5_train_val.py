import os
import shutil
import random
ratio=0.1
img_dir='./yolov5/slice/JPEGImages'
label_dir='./yolov5/slice/worktxt'
train_img_dir='./yolov5/VOC_v5/images/train'
val_img_dir='./yolov5/VOC_v5/images/val'
train_label_dir='./yolov5/VOC_v5/labels/train'
val_label_dir='./yolov5/VOC_v5/labels/val'
if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)
if not os.path.exists(val_img_dir):
    os.makedirs(val_img_dir)
if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)
if not os.path.exists(val_label_dir):
    os.makedirs(val_label_dir)
names=os.listdir(img_dir)
val_names=random.sample(names,int(len(names)*ratio))

cnt_1=0
cnt_2=0
for name in names:
    if name in val_names:
        #cnt_1+=1
        #if cnt_1>100:
            #break
        shutil.copy(os.path.join(img_dir,name),os.path.join(val_img_dir,name))
        shutil.copy(os.path.join(label_dir, name[:-4]+'.txt'), os.path.join(val_label_dir, name[:-4]+'.txt'))
    else:
        #cnt_2+=1
        #if cnt_2>1000:
            #break
        shutil.copy(os.path.join(img_dir, name), os.path.join(train_img_dir, name))
        shutil.copy(os.path.join(label_dir, name[:-4] + '.txt'), os.path.join(train_label_dir, name[:-4] + '.txt'))