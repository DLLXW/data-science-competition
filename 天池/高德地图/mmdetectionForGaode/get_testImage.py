import os
import shutil
images_dir='data/VOC2007/JPEGImages'
txt_dir='data/VOC2007/ImageSets/Main/val.txt'
save_dir='data/demo_test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
r=open(txt_dir,'r')
lines=r.readlines()
for i in range(len(lines)):
    line=lines[i][:-1]
    shutil.copy(os.path.join(images_dir,line+'.jpg'),os.path.join(save_dir,line+'.jpg'))
r.close()