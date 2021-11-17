import numpy as np
import shutil
import os
import cv2
'''
利用yolov5x对400张未标注的图做伪标签，用于BSN模型的训练
'''
def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
def deal_txt(txt_path):
    with open(os.path.join(txts360_dir,txt_path),'r') as f:
        lines=f.readlines()
    image=cv2.imread(os.path.join(images360_dir,txt_path.replace('txt','jpg')))
    ht,wd,_=image.shape
    points=[]
    for line in lines:
        line=line.strip('\n')
        line=[float(p) for p in line.split(' ')]
        center_x=line[1]*wd
        center_y=line[2]*ht
        dis=max(line[3]*wd,line[4]*ht)
        points.append([center_x,center_y,dis]) 
    np.save(os.path.join(npys450_dir,txt_path.replace('txt','npy')),points)
    shutil.copy(os.path.join(images360_dir,txt_path.replace('txt','jpg')),os.path.join(images450_dir,txt_path.replace('txt','jpg')))
    # for point in points:
    #         cv2.circle(image,(int(point[0]),int(point[1])),radius=10,color=(0,0,255),thickness=2)
    # cv2.imwrite('./pesudo_test.jpg',image)
    return points
if __name__=="__main__":
    images360_dir='../../data/400/images'
    txts360_dir='../../detect/yolov5-master/runs/detect/exp52/labels'
    #images90_dir='../../data/90/images'
    #npys90_dir='../../data/90/point_npy'
    images450_dir='../../data/450/images'
    npys450_dir='../../data/450/point_npy'
    #
    make_dir(images450_dir)
    make_dir(npys450_dir)
    #
    txt_paths=os.listdir(txts360_dir)
    for txt_path in txt_paths:
        points=deal_txt(txt_path)
        #print(points)
        #break

