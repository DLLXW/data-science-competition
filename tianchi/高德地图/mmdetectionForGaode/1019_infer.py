from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import glob
import torch
import numpy as np
import cv2
import pandas as pd
import time
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support
#
import json

import os
import timm
import torch
import torch.optim as optim
from torch import nn
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from timm.data import transforms_factory
from timm.optim import optim_factory
from timm.scheduler import scheduler_factory
import pandas as pd
import numpy as np
import glob

#test_image_path = '/tcdata/amap_traffic_final_test_data/'
#test_json = '/tcdata/amap_traffic_final_test_0906.json'
test_image_path = '/home/admins/qyl/gaode/raw_data/amap_traffic_b_test_0828/'
test_json = '/home/admins/qyl/gaode/raw_data/amap_traffic_annotations_b_test_0828.json'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
TRAIN_BATCH_SIZE = 4


class roadDatasetInfer(Dataset):
    def __init__(self, data_dir):
        self.paths = sorted(glob.glob(data_dir + '/*/*'))
        self.data_transforms = A.Compose([
            A.Resize(height=500, width=900),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
            A.Normalize(mean=(0.446, 0.469, 0.472), std=(0.326, 0.330, 0.338), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])

    def __getitem__(self, index):
        sample_path = self.paths[index]
        img = Image.open(sample_path)
        img = img.convert('RGB')
        img = np.array(img)
        img = self.data_transforms(image=img)['image']
        return img, sample_path

    def __len__(self):
        return len(self.paths)

#获取2048维特征
def get_pool_frt(model):
    model.eval()
    path_list = []
    temp_list = []
    temp_dict = {}
    save_dict = {}
    all_dict = {}
    sun_set = roadDatasetInfer(test_image_path)
    device = torch.device("cuda")
    data_loaders = DataLoader(sun_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=4)
    for data in data_loaders:
        inputs, paths = data

        inputs = inputs.to(device)
        with torch.no_grad():
            temp = model.forward_features(inputs)
            output = model.global_pool(temp)
        output = output.data.cpu().detach().numpy().tolist()
        # temp = temp.cpu().detach().numpy().tolist()
        # print(paths)
        # print(output.shape)
        # assert False
        for idx in range(len(output)):
            temp_dict[paths[idx].split('/')[-2] + '_' + paths[idx].split('/')[-1]] = output[idx]
    with open(test_json) as f:
        submit = json.load(f)
    submit_annos = submit['annotations']
    for i in range(len(submit_annos)):
        submit_anno = submit_annos[i]
        imgId = submit_anno['id']
        key_frame = submit_anno['key_frame']
        all_res = []
        for name in submit_anno['frames']:
            name_pre = temp_dict[imgId + '_' + name['frame_name']]
            all_res.append(name_pre)
        all_dict[imgId] = np.array(all_res).mean(axis=0)
        save_dict[imgId] = temp_dict[imgId + '_' + key_frame]

    all_df = pd.DataFrame.from_dict(all_dict, orient='index', columns=['F{}'.format(i) for i in range(2048)])
    all_df = all_df.reset_index().rename(columns={'index': 'map_id'})
    #all_df.to_csv('final_feature2.csv', index=False)
    return all_df



#数车特征
def get_frt(img,img_name):
    #
    detections = inference_detector(model_car, img)
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

#封闭类置信度特征
def get_frt_fb(img, img_name):
    detections = inference_detector(model_fb, img)
    detections = detections[0]
    scores = detections[:, 4]
    if len(scores) > 0:
        max_score = np.max(scores)  # 把最大置信度认为赛第四类的概率
        cnt = 0
        for j in range(len(detections)):
            score = scores[j]
            if score > 0.7:  # 认为大于0.7的肯定大概率是第四类
                cnt += 1
        if max_score > 0.9:  # 大于0.9直接认为是第四类
            is_fb = 3
        else:
            is_fb = -1
    else:
        max_score = 0
        is_fb = -1
        cnt = 0
    features_dic_fb['name'].append(img_name)
    features_dic_fb['fb_cnt'].append(cnt)
    features_dic_fb['fb_prob'].append(max_score)
    features_dic_fb['is_fb'].append(is_fb)


#
def eval_score(y_test, y_pre):
    _, _, f_class, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pre, labels=[0, 1, 2, 3],
                                                       average=None)
    fper_class = {'畅通': f_class[0], '缓行': f_class[1], '拥堵': f_class[2], '封闭': f_class[3]}
    weight_score = 0.1 * f_class[0] + 0.2 * f_class[1] + 0.3 * f_class[2] + 0.4 * f_class[3]
    return weight_score


#
def count(path):
    count_result = {'CT': 0, 'HX': 0, 'YD': 0, 'FB': 0}
    with open(path) as f:
        submit = json.load(f)
    submit_annos = submit['annotations']
    submit_result = []
    for i in range(len(submit_annos)):
        submit_anno = submit_annos[i]
        status = submit_anno['status']
        submit_result.append(status)
        if status == 0:
            count_result['CT'] += 1
        elif status == 1:
            count_result['HX'] += 1
        elif status == 2:
            count_result['YD'] += 1
        else:
            count_result['FB'] += 1

    return count_result

def groupby_detectFrt(df):
    #df=data.copy()
    map_id_list = [int(name.split('_')[0]) for name in df['name'].values]
    df['map_id'] = map_id_list
    df.drop(['name'], axis=1, inplace=True)
    rawFrtColumns=df.drop(['map_id'],axis=1).columns
    frt_name=["std",'sum','max','min','median']
    df = df.groupby("map_id").agg(frt_name).reset_index()
    df.columns =['map_id'] + ['{}_{}'.format(i,j) for i in rawFrtColumns for j in frt_name]

    return df
if __name__ == '__main__':
    #
    begin_time = time.time()
    features_name = ['name',
                     'car_cnt',
                     'car_scaleSum', 'car_scaleMean', 'car_scaleMax',
                     'car_xSum', 'car_xMean', 'car_xMax', 'car_xMin',
                     'car_ySum', 'car_yMean', 'car_yMax', 'car_yMin',
                     'car_disSum', 'car_dis',
                     'x_disSum', 'x_disMean', 'x_disMin', 'x_disMax',
                     'y_disSum', 'y_disMean', 'y_disMin', 'y_disMax',
                     'roi_car_cnt', 'roi_car_sizeSum', 'roi_car_size', 'roi_car_sizeMax', 'roi_car_sizeMin',
                     'roi_car_sizeStd',
                     'roi_car_cnt1', 'roi_car_sizeSum1', 'roi_car_size1', 'roi_car_sizeMax1', 'roi_car_sizeMin1',
                     'roi_car_sizeStd1',
                     'roi_car_cnt2', 'roi_car_sizeSum2', 'roi_car_size2', 'roi_car_sizeMax2', 'roi_car_sizeMin2',
                     'roi_car_sizeStd2',
                     'roi_car_cnt3', 'roi_car_sizeSum3', 'roi_car_size3', 'roi_car_sizeMax3', 'roi_car_sizeMin3',
                     'roi_car_sizeStd3',
                     'roi_car_cnt4', 'roi_car_sizeSum4', 'roi_car_size4', 'roi_car_sizeMax4', 'roi_car_sizeMin4',
                     'roi_car_sizeStd4',
                     ]
    features_dic = {}
    fill_null = 0  # 对于空值使用0来填充
    for fea in features_name:
        features_dic[fea] = []

    config = 'configs_raw/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    checkpoint = 'checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
    # build the model from a config file and a checkpoint file
    model_car = init_detector(config, checkpoint)
    # images
    img_paths = sorted(glob.glob(test_image_path + '/*/*'))
    print('data size:', len(img_paths))
    # demo()
    cnt_programe = 0
    for img_path in img_paths:
        cnt_programe += 1
        print(img_path, ' ', cnt_programe)
        seq = img_path.split('/')[-2]
        frame = img_path.split('/')[-1]
        get_frt(img_path, img_name=seq + '_' + frame)
    #
    df_car = pd.DataFrame(features_dic)  # 这里的df表示的是检测特征(车的数量，大小，位置等特征)
    print('数车特征制作完毕.....特征shape：', df_car.shape)
    # -------------------分割线------------------
    # 下面的代码用于计算针对封闭类的检测特征
    #CONFIG_FILE_fb = '/workspace/docker/gaode/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_gaode.py'
    #CHECKPOINT_PATH_fb = '/workspace/docker/gaode/work_dirs/cascade_rcnn_r50_fpn_1x_gaode/latest.pth'
    CONFIG_FILE_fb = './configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_gaode.py'
    CHECKPOINT_PATH_fb = './work_dirs/cascade_rcnn_r50_fpn_1x_gaode/latest.pth'
    #
    model_fb = init_detector(CONFIG_FILE_fb, CHECKPOINT_PATH_fb)
    features_dic_fb = {'name': [], 'fb_cnt': [], 'fb_prob': [], 'is_fb': []}  # 把检测得到的最大置信度的score当作特征,用于决定第四类(封闭类)
    cnt_programe = 0
    for img_path in img_paths:
        cnt_programe += 1
        print(img_path, ' ', cnt_programe)
        seq = img_path.split('/')[-2]
        frame = img_path.split('/')[-1]
        get_frt_fb(img_path, img_name=seq + '_' + frame)
    #
    df_fb = pd.DataFrame(features_dic_fb)
    print('封闭类特征制作完毕.........特征shape:', df_fb.shape)
    # -------------------分割线------------------
    # -------下面的代码是利用分类来制作概率特征--------------------
    df_pool_frt = get_pool_frt()

    # 三类特征进行merge操作,得到最终的测试集特征
    test_df = df_car.merge(df_fb)
    test_df = test_df.fillna(0)#填空值
    test_df = groupby_detectFrt(test_df)#按照序列号进行合并
    test_df = test_df.merge(df_pool_frt)#合并上分类pool特征
    test_data = test_df.drop(['map_id'], axis=1)
    print('test_data shape：', test_data.shape)
    # 读取训练集合特征
    train_df = pd.read_csv('mm_mm_cls_frts.csv')
    kind = train_df['label']
    train_data=train_df.drop(['map_id','label'],axis=1)
    #
    llf = lgb.LGBMClassifier(num_leaves=9
                             , max_depth=11
                             , learning_rate=0.2
                             , n_estimators=200
                             , objective='multiclass'
                             , n_jobs=8
                             , reg_alpha=0
                             , reg_lambda=0
                             )
    #
    #
    answersLgb = []
    mean_f1 = 0
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    for train, test in sk.split(train_data, kind):
        x_train = train_data.iloc[train]
        y_train = kind.iloc[train]
        x_test = train_data.iloc[test]
        y_test = kind.iloc[test]

        llf.fit(x_train, y_train)
        pred_lgb = llf.predict(x_test)
        weight_lgb = eval_score(y_test, pred_lgb)

        prob_lgb = llf.predict_proba(x_test)
        mean_f1 += weight_lgb / n_splits
        test_lgb = llf.predict_proba(test_data)
        ans = test_lgb
        answersLgb.append(np.argmax(ans, axis=1))
    print('mean weighted f1:', mean_f1)
    #
    fina = []
    for i in range(len(test_data)):
        counts = np.bincount(np.array(answersLgb, dtype='int')[:, i])
        fina.append(np.argmax(counts))
    #
    pre_dict = {}
    test_names = test_df['map_id'].values
    for i in range(len(fina)):
        pre_dict[test_names[i]] = fina[i]
    # some parameters
    count_result = {'x': 0, 'y': 0, 'z': 0, "w": 0}
    with open(test_json) as f:
        submit = json.load(f)
    submit_annos = submit['annotations']
    submit_result = []
    for i in range(len(submit_annos)):
        submit_anno = submit_annos[i]
        imgId = submit_anno['id']
        #frame_name = [imgId + '_' + i['frame_name'] for i in submit_anno['frames']]
        #status_all = [pre_dict[i] for i in frame_name]
        #status = max(status_all, key=status_all.count)
        status=pre_dict[int(imgId)]
        submit['annotations'][i]['status'] = status

    submit_json = '/workspace/result.json'
    json_data = json.dumps(submit)
    with open(submit_json, 'w') as w:
        w.write(json_data)
    print(count(submit_json))

