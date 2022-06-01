import os
import glob
train_txt=open('./dataset/train.txt','w')
train_val_txt=open('./dataset/train_val.txt','w')
val_txt=open('./dataset/val.txt','w')
img_paths=sorted(glob.glob('/home/trojanjet/3d_face/data/train/*/*/*/*/*'))
new_img_paths=[]
for p in img_paths:
    if p[-5:]=='t.jpg':
        new_img_paths.append(p)
#
val_len=int(0.1*len(new_img_paths))
for i in range(0,len(new_img_paths),3):
    if i>val_len:
        train_txt.write(new_img_paths[i]+' '+new_img_paths[i+1]+' '+new_img_paths[i+2]+'\n')
    else:
        val_txt.write(new_img_paths[i]+' '+new_img_paths[i+1]+' '+new_img_paths[i+2]+'\n')
    train_val_txt.write(new_img_paths[i]+' '+new_img_paths[i+1]+' '+new_img_paths[i+2]+'\n')
train_txt.close()
train_val_txt.close()
val_txt.close()