'''
制作voc数据集的训练集/验证集
'''
import os
import random
import sys

use_slice=True
root_path = './yolov5/slice'

xmlfilepath = root_path + '/Annotations'

txtsavepath = root_path + '/ImageSets/Main'


if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

trainval_percent = 1
train_percent = 0.9
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
if use_slice:
    train=[]
    for i in range(num):
        per_name=total_xml[i]
        if int(per_name.split('|')[0])%10!=0:
            train.append(i)

print("train and val size:", tv)
print("train size:", len(train))

ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
