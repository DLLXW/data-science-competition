import os
import cv2
import json
import numpy as np

def dea_boarder(bbox):
    xmin,ymin,xmax,ymax,wd,ht=bbox
    xmax=np.clip(xmax, 0, wd)
    ymax=np.clip(ymax, 0, ht)
    xmin=np.clip(xmin, 0, xmax)
    ymin=np.clip(ymin, 0, ymax)
    bbox=xmin,ymin,xmax,ymax,wd,ht
    return bbox
#
root_dir_img='../../data/140/images'
root_dir='../../data/140/annotations'
save_dir_bbox='../../data/140/bbox_npy/'
save_dir_point='../../data/140/point_npy/'
if not os.path.exists(save_dir_bbox):os.makedirs(save_dir_bbox)
if not os.path.exists(save_dir_point):os.makedirs(save_dir_point)
anns=os.listdir(root_dir)
drop=0
for ann in anns:
    img_show=cv2.imread(os.path.join(root_dir_img,ann.replace('json','jpg')))
    ht,wd,_=img_show.shape
    with open(os.path.join(root_dir,ann)) as f:
        data=json.load(f)
    #print(ann)
    data=data["shapes"]
    labels_bbox=[]
    labels_point=[]
    for per in data:
        if per['shape_type']=='polygon':#标注的是关键点轮廓，求xy平均即可当作中心点
            #print('polygon',ann)
            polygon=np.array(per["points"])
            xmin=np.min(polygon[:,0])
            ymin=np.min(polygon[:,1])
            xmax=np.max(polygon[:,0])
            ymax=np.max(polygon[:,1])
            w=xmax-xmin
            h=ymax-ymin
            x_center=(xmin+xmax)/2
            y_center=(ymin+ymax)/2
            dis=max(xmax-xmin,ymax-ymin)
            if xmin<xmax and ymin<ymax and xmin>0 and xmax<wd and ymin>0 and ymax<ht:
                labels_bbox.append([xmin,ymin,w,h,wd,ht])
                labels_point.append([x_center,y_center,dis])
            else:
                drop+=1
        else:#标注的是框坐标，计算得到中心点即可
            '''
            因为标注的位置可能是左下、右上，所以需要转换
            xmin=min(x1,x2)
            xmax=max(x1,x2)
            ymin=min(y1,y2)
            ymax=max(y1,y2)
            '''
            x1=per["points"][0][0]
            y1=per["points"][0][1]
            x2=per["points"][1][0]
            y2=per["points"][1][1]
            xmin=min(x1,x2)
            xmax=max(x1,x2)
            ymin=min(y1,y2)
            ymax=max(y1,y2)
            w=xmax-xmin
            h=ymax-ymin
            if xmin<xmax and ymin<ymax and xmin>0 and xmax<wd and ymin>0 and ymax<ht:
                x_center=(xmin+xmax)/2
                y_center=(ymin+ymax)/2
                dis=max(xmax-xmin,ymax-ymin)
                labels_bbox.append([xmin,ymin,w,h,wd,ht])
                labels_point.append([x_center,y_center,dis])
            else:
                bbox=[xmin,ymin,xmax,ymax,wd,ht]
                bbox=dea_boarder(bbox)
                xmin,ymin,xmax,ymax,wd,ht=bbox
                assert xmin<xmax and ymin<ymax and xmin>=0 and xmax<=wd and ymin>=0 and ymax<=ht
                w=xmax-xmin
                h=ymax-ymin
                x_center=(xmin+xmax)/2
                y_center=(ymin+ymax)/2
                dis=max(xmax-xmin,ymax-ymin)
                labels_bbox.append([xmin,ymin,w,h,wd,ht])
                labels_point.append([x_center,y_center,dis])
                #cv2.circle(img_show,(int(xmin),int(ymin)),radius=15,color=(0,0,255),thickness=2)
                #cv2.circle(img_show,(int(xmax),int(ymax)),radius=15,color=(0,0,255),thickness=2)
                #drop+=1
    labels_bbox=np.array(labels_bbox)
    labels_point=np.array(labels_point)
    #print(labels_bbox.shape)
    np.save(os.path.join(save_dir_bbox,ann.replace('json','npy')),labels_bbox)
    np.save(os.path.join(save_dir_point,ann.replace('json','npy')),labels_point)
    #cv2.imwrite('./demo_debug_cc.jpg',img_show)
    #break
#print(drop)