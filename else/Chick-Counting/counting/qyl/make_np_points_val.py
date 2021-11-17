'''
为手动标的40张做处理，转为np格式
'''
import os
import cv2
import json
import numpy as np
import pandas as pd
root_dir_img='../../data/ann_test/images'
root_dir='../../data/ann_test/annotations'
save_dir='../../data/ann_test/point_npy/'
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
        center_x=per["points"][0][0]
        center_y=per["points"][0][1]
        #if center_x>0 and center_x<wd and center_y>0 and center_y<ht:
            #labels.append([center_x,center_y])
        labels.append([center_x,center_y])
    labels=np.array(labels)
    np.save(os.path.join(save_dir,ann.replace('json','npy')),labels)
    #break

label_csv='../../data/val_df_20.csv'
df=pd.DataFrame(columns=['image_name'])
# ##-------------------写入csv-------------------
df['image_name']=os.listdir(root_dir_img)
df.to_csv(label_csv,index=False)
print(df)