import os
import cv2
import json
import numpy as np
root_dir_img='../../data/100/images'
root_dir='../../data/100/annotations'
save_dir='../../data/100/point_npy_2dim/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
anns=os.listdir(root_dir)
pp=[]
for ann in anns:
    #ann='20210815_20340107064603_20340107070829_203801.mp4-98a9c3d55d6fadb67274b04a91f8476f.json'
    img_show=cv2.imread(os.path.join(root_dir_img,ann.replace('json','jpg')))
    ht,wd,_=img_show.shape
    with open(os.path.join(root_dir,ann)) as f:
        data=json.load(f)
    #print(ann)
    data=data["shapes"]
    labels=[]
    for per in data:
        if per['shape_type']=='polygon':#标注的是关键点轮廓，求xy平均即可当作中心点
            polygon=np.array(per["points"])
            xmin=np.min(polygon[:,0])
            ymin=np.min(polygon[:,1])
            xmax=np.max(polygon[:,0])
            ymax=np.max(polygon[:,1])
            polygon=np.mean(polygon,axis=0)
            dis=max(xmax-xmin,ymax-ymin)
            if polygon[0]>0 and polygon[0]<wd and polygon[1]>0 and polygon[1]<ht:
                polygon=polygon.tolist()
                #polygon.append(dis)
            labels.append(polygon)
        else:#标注的是框坐标，计算得到中心点即可
            x1=per["points"][0][0]
            y1=per["points"][0][1]
            x2=per["points"][1][0]
            y2=per["points"][1][1]
            center_x=(x1+x2)/2
            center_y=(y1+y2)/2
            dis=max(x2-x1,y2-y1)
            #if center_x>0 and center_x<wd and center_y>0 and center_y<ht:
            labels.append([center_x,center_y])
    labels=np.array(labels)
    pp.append(len(labels))
    print(labels.shape)
    #np.save(os.path.join(save_dir,ann.replace('json','npy')),labels)
    #break
pp=sorted(pp)
print(pp)
print(np.median(pp))
'''
[1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 167, 168, 170, 171, 171, 172, 172, 173, 173, 173, 175, 175, 175, 
175, 175, 176, 177, 180, 181, 181, 183, 184, 184, 184, 184, 186, 187, 188, 189, 189, 192, 192, 193, 193,
 195, 196, 196, 205, 206, 208, 210, 211, 211, 211, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 216, 216, 216, 216, 216, 217, 217, 217, 
217, 217, 217, 217, 218, 219, 219, 219, 219, 220, 220, 221, 221, 222, 222, 222, 223, 223, 224, 224, 225, 225, 226, 226, 226, 227, 233, 233]
'''
