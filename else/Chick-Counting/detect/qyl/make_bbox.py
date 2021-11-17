import os
import cv2
import json
import numpy as np
root_dir_img='../../rare_sample/images'
root_dir='../../rare_sample/annotations'
save_dir='../../rare_sample/bbox_npy/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
anns=os.listdir(root_dir)
for ann in anns:
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
            w=xmax-xmin
            h=ymax-ymin
            if xmin<xmax and ymin<ymax and xmin>0 and xmax<wd and ymin>0 and ymax<ht:
                labels.append([xmin,ymin,w,h,wd,ht])
        else:#标注的是框坐标，计算得到中心点即可
            xmin=per["points"][0][0]
            ymin=per["points"][0][1]
            xmax=per["points"][1][0]
            ymax=per["points"][1][1]
            w=xmax-xmin
            h=ymax-ymin
            if xmin<xmax and ymin<ymax and xmin>0 and xmax<wd and ymin>0 and ymax<ht:
                labels.append([xmin,ymin,w,h,wd,ht])
    labels=np.array(labels)
    print(labels.shape)
    np.save(os.path.join(save_dir,ann.replace('json','npy')),labels)
    #break
