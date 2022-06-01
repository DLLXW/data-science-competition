import json
import cv2
from ensemble_boxes import *
import os
from tqdm import tqdm
import glob
result_dir_1='./just05(1).json'
result_dir_2='./r2_0_3.json'
with open(result_dir_1,'r') as f:
    data_1=json.load(f)
#
with open(result_dir_2,'r') as f:
    data_2=json.load(f)
#
image_dic={}
for per_path in tqdm(glob.glob('./testB_imgs/*')):
    name=per_path.split('/')[-1]
    image_dic[name]=cv2.imread(per_path,cv2.IMREAD_GRAYSCALE).shape[:2]
    #print(image_dic)
#
with open('dic_name.json','w') as f:
     json.dump(image_dic,f)

with open('dic_name.json','r') as f:
    image_dic=josn.load(f)

#
submit_result=[]
for key in image_dic.keys():
    boxes_list_1 =[]
    scores_list_1=[]
    labels_list_1=[]
    boxes_list_2 =[]
    scores_list_2=[]
    labels_list_2=[]
    #model 1
    win_h,win_w=image_dic[key]
    for per_pre in data_1:
        name=per_pre['name']
        if name==key:
            bbox=per_pre['bbox']
            score=per_pre['score']
            #
            #
            boxes_list_1.append([bbox[0]/win_w,bbox[1]/win_h,bbox[2]/win_w,bbox[3]/win_h])
            scores_list_1.append(score)
            labels_list_1.append(per_pre['category'])
    #model 2
    for per_pre in data_2:
        name=per_pre['name']
        if name==key:
            bbox=per_pre['bbox']
            score=per_pre['score']
            #
            boxes_list_2.append([bbox[0]/win_w,bbox[1]/win_h,bbox[2]/win_w,bbox[3]/win_h])
            scores_list_2.append(score)
            labels_list_2.append(per_pre['category'])
    
    weights = [1, 1]
    iou_thr = 0.3
    skip_box_thr = 0.0001
    sigma = 0.1
    boxes_list=[boxes_list_1,boxes_list_2]
    scores_list=[scores_list_1,scores_list_2]
    labels_list=[labels_list_1,labels_list_2]
    #
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
             weights=[1,1], iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    #

    for i in range(len(boxes)):
        box=boxes[i]
        box=[int(box[0]*win_w),int(box[1] * win_h),int(box[2] * win_w),int(box[3] * win_h)]
        submit_result.append(
            {'name': key, 'category': int(labels[i]), 'bbox': box, 'score': float(str(scores[i])[:5])})
#
with open('results/resut_wbf.json', 'w') as fp:
    json.dump(submit_result, fp, indent=4, ensure_ascii=False)

with open('dic_name.json','w') as f:
     json.dump(image_dic,f)