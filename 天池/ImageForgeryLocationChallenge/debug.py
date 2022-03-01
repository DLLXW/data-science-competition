import cv2.cv2 as cv
import cv2
import os

image=cv2.imread('/home/trojanjet/baidu_qyl/yaogan/ctseg7/train/dataset/val_images/0_27065.tif',cv2.IMREAD_UNCHANGED)
mask = cv2.imread('/home/trojanjet/baidu_qyl/yaogan/ctseg7/train/dataset/val_labels/2_827.png', cv2.IMREAD_GRAYSCALE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
dic=[0]*47
print(mask.shape)
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        dic[mask[i][j]]+=1
print(dic)