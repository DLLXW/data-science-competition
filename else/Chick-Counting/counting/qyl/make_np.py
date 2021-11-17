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
for ann in anns:
    #ann='20210716_20340101015744_20340101023436_182240.mp4-0b591cb612585b3c1979c21c83aa55ec.json'
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
            if center_x>0 and center_x<wd and center_y>0 and center_y<ht:
                labels.append([center_x,center_y])
    labels=np.array(labels)
    np.save(os.path.join(save_dir,ann.replace('json','npy')),labels)
    #break
    
