import os
import glob
import numpy as np
from config import Config

print("make label .txt and split train/val set for datasets")
opt=Config()
image_paths=glob.glob(opt.trainValConcat_dir+'/*/*')
image_paths=np.random.permutation(image_paths).tolist()
data_size=len(image_paths)
w1=open(opt.train_list,'w')
w2=open(opt.val_list,'w')
val_ratio=0.1
print('data_size:{}, train_size:{}, test_size:{}'.format(data_size,int(data_size*(1-val_ratio)),int(data_size*val_ratio)))
train_str=''
val_str=''
for i in range(data_size):
    img_path=image_paths[i].split('/')
    name=img_path[-1]
    cls=img_path[-2]
    if i%int(1/val_ratio)==0:
        val_str+=cls+'/'+name+','+cls+'\n'
    else:
        train_str += cls+'/'+name + ',' + cls + '\n'
w1.write(train_str[:-1])
w2.write(val_str[:-1])
w1.close()
w2.close()
