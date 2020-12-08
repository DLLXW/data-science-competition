from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import glob
import torch
import numpy as np
import cv2
import pandas as pd
import time
def demo():
    img = './demo/000492_1.jpg'
    result = inference_detector(model, img)
    print(result[1],type(result[1]),len(result[1]))
    print(result[1]==[])
    #print(len(result))
    show_result_pyplot(model, img, result, score_thr=0.3)
def get_frt(img,img_name):
    #
    detections = inference_detector(model, img)
    detections=detections[2]#取出car这个类别
    image=cv2.imread(img)
    car_cnt = len(detections)#这张图片中车的数量
    raw_h, raw_w, _ = image.shape
    img_scale=raw_h*raw_w
    roi_scale = []#roi区域的车的大小
    roi_car_cnt = 0#roi区域的车的数量
    roi_scale1 = []  # roi区域的车的大小
    roi_car_cnt1 = 0  # roi区域的车的数量
    roi_scale2 = []  # roi区域的车的大小
    roi_car_cnt2 = 0  # roi区域的车的数量
    roi_scale3 = []  # roi区域的车的大小
    roi_car_cnt3 = 0  # roi区域的车的数量
    roi_scale4 = []  # roi区域的车的大小
    roi_car_cnt4 = 0  # roi区域的车的数量

    if car_cnt > 0:
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        w = x2-x1
        h = y2-y1
        # scores = detections[:, 4]
        c_x=(x1+x2)/2/raw_w
        c_y=(y1+y2)/2/raw_h
        w_x_h = (x2 - x1) * (y2 - y1) / img_scale
        x_dis=c_x-0.5
        y_dis=1-c_y
        c_dis=np.sqrt((c_x-0.5)**2+(c_y-0.5)**2)
        #中心横/纵坐标分布特征
        f_c_xSum = np.sum(c_x)
        f_c_xMean = np.mean(c_x)
        f_c_xMin = np.min(c_x)
        f_c_xMax = np.max(c_x)
        f_c_ySum = np.sum(c_y)
        f_c_yMean = np.mean(c_y)
        f_c_yMin = np.min(c_y)
        f_c_yMax = np.max(c_y)
        #横坐标离中心点的距离/纵坐标离底部的距离特征
        f_x_disSum = np.sum(x_dis)
        f_x_disMean=np.mean(x_dis)
        f_x_disMin= np.min(x_dis)
        f_x_disMax = np.max(y_dis)
        #
        f_y_disSum = np.sum(y_dis)
        f_y_disMean = np.mean(y_dis)
        f_y_disMin = np.min(y_dis)
        f_y_disMax = np.max(y_dis)
        #
        f_c_disSum=np.sum(c_dis)
        f_c_disMean = np.mean(c_dis)
        #车的大小特征
        f_scaleSum = np.sum(w_x_h)
        f_scaleMean = np.mean(w_x_h)
        f_scaleMax = np.max(w_x_h)

        for j in range(car_cnt):
            x = (x1[j] + x2[j]) / 2
            y = (y1[j] + y2[j]) / 2
            w_obj = w[j]
            h_obj = h[j]
            if raw_w/4<x<3*raw_w/4:#roi区域
                roi_scale.append(w_obj*h_obj/img_scale)
                roi_car_cnt+=1
            #
            if x>raw_w/4:#roi1区域
                roi_scale1.append(w_obj*h_obj/img_scale)
                roi_car_cnt1+=1
            if x<3*raw_w/4:#roi2区域
                roi_scale2.append(w_obj*h_obj/img_scale)
                roi_car_cnt2+=1

            if x>raw_w/4 and y>raw_h/2:#roi3区域
                roi_scale3.append(w_obj*h_obj/img_scale)
                roi_car_cnt3+=1
            if x<3*raw_w/4 and y>raw_h/2:#roi4区域
                roi_scale4.append(w_obj*h_obj/img_scale)
                roi_car_cnt4+=1

    else:
        # 中心横/纵坐标分布特征
        f_c_xSum = fill_null
        f_c_xMean = fill_null
        f_c_xMin = fill_null
        f_c_xMax = fill_null
        f_c_ySum = fill_null
        f_c_yMean = fill_null
        f_c_yMin = fill_null
        f_c_yMax = fill_null
        # 横坐标离中心点的距离/纵坐标离底部的距离特征
        f_x_disSum = 1
        f_x_disMean = 1
        f_x_disMin = 1
        f_x_disMax = 1
        #
        f_y_disSum = 1
        f_y_disMean = 1
        f_y_disMin = 1
        f_y_disMax = 1
        #
        f_c_disSum = 1
        f_c_disMean = 1
        # 车的大小特征
        f_scaleSum = fill_null
        f_scaleMean = fill_null
        f_scaleMax = fill_null
    if roi_scale != []:  # 对于空值使用0来填充
        f_roiareaSum = np.sum(roi_scale)
        f_roiarea = np.mean(roi_scale)
        f_roiareaMax = np.max(roi_scale)
        f_roiareaMin = np.min(roi_scale)
        f_roiareaStd = np.nanstd(roi_scale)
    else:
        f_roiareaSum = fill_null
        f_roiarea = fill_null
        f_roiareaMax = fill_null
        f_roiareaMin = fill_null
        f_roiareaStd = fill_null
    if roi_scale1 != []:  # 对于空值使用0来填充
        f_roiareaSum1 = np.sum(roi_scale1)
        f_roiarea1 = np.mean(roi_scale1)
        f_roiareaMax1 = np.max(roi_scale1)
        f_roiareaMin1 = np.min(roi_scale1)
        f_roiareaStd1 = np.nanstd(roi_scale1)
    else:
        f_roiareaSum1 = fill_null
        f_roiarea1 = fill_null
        f_roiareaMax1 = fill_null
        f_roiareaMin1 = fill_null
        f_roiareaStd1 = fill_null
    if roi_scale2 != []:  # 对于空值使用0来填充
        f_roiareaSum2 = np.sum(roi_scale2)
        f_roiarea2 = np.mean(roi_scale2)
        f_roiareaMax2 = np.max(roi_scale2)
        f_roiareaMin2 = np.min(roi_scale2)
        f_roiareaStd2 = np.nanstd(roi_scale2)
    else:
        f_roiareaSum2 = fill_null
        f_roiarea2 = fill_null
        f_roiareaMax2 = fill_null
        f_roiareaMin2 = fill_null
        f_roiareaStd2 =fill_null
    if roi_scale3 != []:  # 对于空值使用0来填充
        f_roiareaSum3 = np.sum(roi_scale3)
        f_roiarea3 = np.mean(roi_scale3)
        f_roiareaMax3 = np.max(roi_scale3)
        f_roiareaMin3 = np.min(roi_scale3)
        f_roiareaStd3 = np.nanstd(roi_scale3)
    else:
        f_roiareaSum3 = fill_null
        f_roiarea3 = fill_null
        f_roiareaMax3 = fill_null
        f_roiareaMin3 = fill_null
        f_roiareaStd3=fill_null
    if roi_scale4 != []:  # 对于空值使用0来填充
        f_roiareaSum4 = np.sum(roi_scale4)
        f_roiarea4 = np.mean(roi_scale4)
        f_roiareaMax4 = np.max(roi_scale4)
        f_roiareaMin4 = np.min(roi_scale4)
        f_roiareaStd4 = np.nanstd(roi_scale3)
    else:
        f_roiareaSum4 = fill_null
        f_roiarea4 = fill_null
        f_roiareaMax4 = fill_null
        f_roiareaMin4 = fill_null
        f_roiareaStd4 =fill_null
        #
    features_dic['name'].append(img_name)
    features_dic['car_cnt'].append(car_cnt)
    features_dic['car_scaleSum'].append(f_scaleSum)
    features_dic['car_scaleMean'].append(f_scaleMean)
    features_dic['car_scaleMax'].append(f_scaleMax)
    #
    features_dic['car_xSum'].append(f_c_xSum)
    features_dic['car_xMean'].append(f_c_xMean)
    features_dic['car_xMax'].append(f_c_xMax)
    features_dic['car_xMin'].append(f_c_xMin)
    #
    features_dic['car_ySum'].append(f_c_ySum)
    features_dic['car_yMean'].append(f_c_yMean)
    features_dic['car_yMax'].append(f_c_yMax)
    features_dic['car_yMin'].append(f_c_yMin)
    #
    features_dic['car_disSum'].append(f_c_disSum)
    features_dic['car_dis'].append(f_c_disMean)
    #
    features_dic['x_disSum'].append(f_x_disSum)
    features_dic['x_disMean'].append(f_x_disMean)
    features_dic['x_disMax'].append(f_x_disMax)
    features_dic['x_disMin'].append(f_x_disMin)
    #
    features_dic['y_disSum'].append(f_y_disSum)
    features_dic['y_disMean'].append(f_y_disMean)
    features_dic['y_disMax'].append(f_y_disMax)
    features_dic['y_disMin'].append(f_y_disMin)
    #
    features_dic['roi_car_cnt'].append(roi_car_cnt)
    features_dic['roi_car_sizeSum'].append(f_roiareaSum)
    features_dic['roi_car_size'].append(f_roiarea)
    features_dic['roi_car_sizeMax'].append(f_roiareaMax)
    features_dic['roi_car_sizeMin'].append(f_roiareaMin)
    features_dic['roi_car_sizeStd'].append(f_roiareaStd)
    #
    features_dic['roi_car_cnt1'].append(roi_car_cnt1)
    features_dic['roi_car_sizeSum1'].append(f_roiareaSum1)
    features_dic['roi_car_size1'].append(f_roiarea1)
    features_dic['roi_car_sizeMax1'].append(f_roiareaMax1)
    features_dic['roi_car_sizeMin1'].append(f_roiareaMin1)
    features_dic['roi_car_sizeStd1'].append(f_roiareaStd1)
    #
    features_dic['roi_car_cnt2'].append(roi_car_cnt2)
    features_dic['roi_car_sizeSum2'].append(f_roiareaSum2)
    features_dic['roi_car_size2'].append(f_roiarea2)
    features_dic['roi_car_sizeMax2'].append(f_roiareaMax2)
    features_dic['roi_car_sizeMin2'].append(f_roiareaMin2)
    features_dic['roi_car_sizeStd2'].append(f_roiareaStd2)
    #
    features_dic['roi_car_cnt3'].append(roi_car_cnt3)
    features_dic['roi_car_sizeSum3'].append(f_roiareaSum3)
    features_dic['roi_car_size3'].append(f_roiarea3)
    features_dic['roi_car_sizeMax3'].append(f_roiareaMax3)
    features_dic['roi_car_sizeMin3'].append(f_roiareaMin3)
    features_dic['roi_car_sizeStd3'].append(f_roiareaStd3)
    #
    features_dic['roi_car_cnt4'].append(roi_car_cnt4)
    features_dic['roi_car_sizeSum4'].append(f_roiareaSum4)
    features_dic['roi_car_size4'].append(f_roiarea4)
    features_dic['roi_car_sizeMax4'].append(f_roiareaMax4)
    features_dic['roi_car_sizeMin4'].append(f_roiareaMin4)
    features_dic['roi_car_sizeStd4'].append(f_roiareaStd4)


    # show the results


if __name__ == '__main__':
    #
    begin_time=time.time()
    features_name = ['name',
                     'car_cnt',
                     'car_scaleSum', 'car_scaleMean', 'car_scaleMax',
                     'car_xSum', 'car_xMean', 'car_xMax', 'car_xMin',
                     'car_ySum', 'car_yMean', 'car_yMax', 'car_yMin',
                     'car_disSum', 'car_dis',
                     'x_disSum', 'x_disMean', 'x_disMin', 'x_disMax',
                     'y_disSum', 'y_disMean', 'y_disMin', 'y_disMax',
                     'roi_car_cnt', 'roi_car_sizeSum', 'roi_car_size', 'roi_car_sizeMax', 'roi_car_sizeMin','roi_car_sizeStd',
                     'roi_car_cnt1', 'roi_car_sizeSum1', 'roi_car_size1', 'roi_car_sizeMax1', 'roi_car_sizeMin1','roi_car_sizeStd1',
                     'roi_car_cnt2', 'roi_car_sizeSum2', 'roi_car_size2', 'roi_car_sizeMax2', 'roi_car_sizeMin2','roi_car_sizeStd2',
                     'roi_car_cnt3', 'roi_car_sizeSum3', 'roi_car_size3', 'roi_car_sizeMax3', 'roi_car_sizeMin3','roi_car_sizeStd3',
                     'roi_car_cnt4', 'roi_car_sizeSum4', 'roi_car_size4', 'roi_car_sizeMax4', 'roi_car_sizeMin4','roi_car_sizeStd4',
                     ]
    features_dic = {}
    fill_null = 0#对于空值使用0来填充
    for fea in features_name:
        features_dic[fea] = []

    train_image_dir='/home/admins/qyl/gaode_classify/dataset/amap_traffic_final_train_data'
    train_json_dir='/home/admins/qyl/gaode_classify/dataset/amap_traffic_final_train_0906.json'
    config = 'configs_raw/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    checkpoint = 'checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=0)
    #images
    img_paths=sorted(glob.glob(train_image_dir+'/*/*'))
    print('data size:',len(img_paths))
    #demo()
    cnt_programe = 0
    for img_path in img_paths:
        cnt_programe+=1
        print(img_path, ' ', cnt_programe)
        seq = img_path.split('/')[-2]
        frame = img_path.split('/')[-1]
        get_frt(img_path, img_name=seq+'_'+frame)
        #if cnt_programe>100:
         #   break
    #
    print(pd.DataFrame(features_dic))
    df=pd.DataFrame(features_dic)
    df.to_csv('train_res2net.csv',index=False)
    print('spend time {}s'.format(time.time()-begin_time))