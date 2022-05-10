import os
'''
Attention:
    After run this code,we need to delete the last '\n' in train.txt or val.txt manually
'''
dataDir='../dataset/label.txt'
trainDir='../dataset/train.txt'
valDir='../dataset/val.txt'
train=open(trainDir,'w')
val=open(valDir,'w')
with open(os.path.join(dataDir), 'r') as fd:
    imgs = fd.readlines()

for i in range(len(imgs)):
    if i%10==0:
        val.write(imgs[i])
    else:
        train.write(imgs[i])
val.close()
train.close()
#类别数统计
from collections import Counter
with open(os.path.join(dataDir), 'r') as fd:
    imgs = fd.readlines()
labels=[int(per.strip('\n').split(',')[-1].strip(' ')) for per in imgs]
c = dict(Counter(labels))
cls=[0]*len(c)
for key in c.keys():
    cls[key]=c[key]
print(cls)


