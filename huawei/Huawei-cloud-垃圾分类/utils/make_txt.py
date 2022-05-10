import os
import cv2
import glob
img_dir="../data/train_data"
names=os.listdir(img_dir)
label_txt='../dataset/label.txt'
w=open(label_txt,'w')
label_str=""
for name in names:
    if name[-4:]=='.txt':
        with open(os.path.join(img_dir,name),'r') as f:
            tmp=f.readlines()[0]+'\n'
        label_str+=tmp
label_str=label_str.strip('\n')
w.write(label_str)
w.close()