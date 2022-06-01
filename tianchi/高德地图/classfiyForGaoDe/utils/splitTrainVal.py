import os
import shutil
import random
'''
Attention:
    After run this code,we need to delete the last '\n' in train.txt or val.txt manually
'''
dataDir='/home/admins/qyl/imageRetrival/train_data/label.txt'
trainDir='../dataset/train.txt'
valDir='../dataset/val.txt'
train=open(trainDir,'w')
val=open(valDir,'w')
with open(os.path.join(dataDir), 'r') as fd:
    imgs = fd.readlines()

for i in range(len(imgs)):
    if i%50==0:
        val.write(imgs[i])
    else:
        train.write(imgs[i])
val.close()
train.close()