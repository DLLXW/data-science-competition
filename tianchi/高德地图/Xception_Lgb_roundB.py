from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cab
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold
import json
import os
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score


def get_keyFrame(mode='train'):
    result = []
    if mode == 'train':
        rawLabelDir = '../user_data/amap_traffic_annotations_train.json'
    else:
        rawLabelDir = '../data/amap_traffic_annotations_b_test_0828.json'

    with open(rawLabelDir) as f:
        d = json.load(f)
    annos = d['annotations']
    for anno in annos:
        imgId = anno['id']
        frame_name = [k['frame_name'] for k in anno['frames']]  # 图片序列
        key_frame = anno['key_frame']
        key_id = imgId + '_' + key_frame
        result.append(key_id)
    return result

def cnt_results(submit):
    count_result = {'畅通': 0, '缓行': 0, '拥堵': 0}
    submit_annos = submit['annotations']
    submit_result = []
    for i in range(len(submit_annos)):
        submit_anno = submit_annos[i]
        status = submit_anno['status']
        submit_result.append(status)
        if status == 0:
            count_result['畅通'] += 1
        elif status == 1:
            count_result['缓行'] += 1
        else:
            count_result['拥堵'] += 1
    return count_result['畅通'], count_result['缓行'], count_result['拥堵']

def process_cnnFrt(df):
    map_id_list = [name[:-4] for name in df['name'].values]
    df['map_id'] = map_id_list
    df.drop(['name'], axis=1, inplace=True)
    return df


# process the detect-features which from single frame
def process_detectYoloFrt(df, mode='train'):
    # df=df[featureyolo]
    cols = list(df.columns)
    yolo = 'yolo'
    df.columns = map(lambda x: str(x) + '_%s' % yolo, cols)
    # df.rename(columns={'name_yolo':'name'},inplace=True)
    # print(df.columns)
    map_id_list = [name.split('_')[0] for name in df['name_yolo'].values]
    df['map_id'] = map_id_list
    df.drop(['name_yolo'], axis=1, inplace=True)
    rawFrtColumns = df.drop(['map_id'], axis=1).columns
    frt_name = ["std", 'median', 'sum', 'max', 'min', 'mean']
    df = df.groupby("map_id").agg(frt_name).reset_index()
    df.columns = ['map_id'] + ['{}_{}'.format(i, j) for i in rawFrtColumns for j in frt_name]
    return df


def process_detectYoloFrtKey(df, mode='train'):
    # df=df[featureyolo]
    cols = list(df.columns)
    yolo = 'yolo_key'
    df.columns = map(lambda x: str(x) + '_%s' % yolo, cols)
    map_id_list = [name.split('_')[0] for name in df['name_yolo_key'].values]
    df['map_id'] = map_id_list
    df.drop(['name_yolo_key'], axis=1, inplace=True)
    return df


def get_data(df):
    map_id_list = []
    label = []
    key_frame_list = []
    jpg_name_1 = []
    jpg_name_2 = []
    gap_time_1 = []
    gap_time_2 = []
    for s in list(df.annotations):
        map_id = s["id"]
        map_key = s["key_frame"]
        frames = s["frames"]
        status = s["status"]
        for i in range(0, len(frames) - 1):  # get per frame
            f = frames[i]
            f_next = frames[i + 1]
            map_id_list.append(map_id)
            key_frame_list.append(map_key)
            jpg_name_1.append(f["frame_name"])
            jpg_name_2.append(f_next["frame_name"])
            gap_time_1.append(f["gps_time"])
            gap_time_2.append(f_next["gps_time"])
            label.append(status)

    train_df = pd.DataFrame({
        "map_id": map_id_list,
        "label": label,
        "key_frame": key_frame_list,
        "jpg_name_1": jpg_name_1,
        "jpg_name_2": jpg_name_2,
        "gap_time_1": gap_time_1,
        "gap_time_2": gap_time_2,

    })
    # print(train_df)

    train_df["gap"] = train_df["gap_time_2"] - train_df["gap_time_1"]
    train_df["gap_time_today"] = train_df["gap_time_1"] % (24 * 3600)
    train_df["hour"] = train_df["gap_time_1"].apply(lambda x: datetime.fromtimestamp(x).hour)
    train_df["minute"] = train_df["gap_time_1"].apply(lambda x: datetime.fromtimestamp(x).minute)
    train_df["day"] = train_df["gap_time_1"].apply(lambda x: datetime.fromtimestamp(x).day)
    train_df["month"] = train_df["gap_time_1"].apply(lambda x: datetime.fromtimestamp(x).month)
    train_df["dayofweek"] = train_df["gap_time_1"].apply(lambda x: datetime.fromtimestamp(x).weekday())

    train_df["key_frame"] = train_df["key_frame"].apply(lambda x: int(x.split(".")[0]))

    train_df = train_df.groupby("map_id").agg({"gap": ["mean", "std", "max", "min"],
                                               "month": ["mean"],
                                               "hour": ["mean"],
                                               "minute": ["mean"],
                                               "dayofweek": ["mean"],
                                               "gap_time_today": ["mean", "std"],
                                               "label": ["mean"],
                                               }).reset_index()
    train_df.columns = ["map_id", "gap_mean", "gap_std", "gap_max", "gap_min", "month_mean",
                        "hour_mean", "minute_mean", "dayofweek_mean", "gap_time_today_mean", "gap_time_today_std",
                        "label"]
    train_df["label"] = train_df["label"].apply(int)

    return train_df

def models():
    xlf = xgb.XGBClassifier(max_depth=7
                            , learning_rate=0.3
                            , n_estimators=100
                            , reg_alpha=0.004
                            , n_jobs=8
                            , importance_type='total_cover'
                            )

    llf = lgb.LGBMClassifier(num_leaves=9
                             , max_depth=6
                             , learning_rate=0.1
                             , n_estimators=60
                             , objective='multiclass'
                             , n_jobs=8
                             , reg_alpha=0
                             , reg_lambda=0
                             )
    clf = cab.CatBoostClassifier(iterations=120
                                 , learning_rate=0.2
                                 , depth=6
                                 , loss_function='MultiClass'
                                 , silent=True
                                 , task_type='GPU'
                                 )
    return xlf,llf,clf

def eval_score(y_test,y_pre):
    _, _, f_class, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pre, labels=[0, 1, 2],
                                                       average=None)
    fper_class = {'畅通': f_class[0], '缓行': f_class[1], '拥堵': f_class[2]}
    weight_f1 = 0.2 * f_class[0] + 0.2 * f_class[1] + 0.6 * f_class[2]
    return weight_f1
#
train_key = get_keyFrame('train')
test_key = get_keyFrame('test')
# features from object detection based on yolov5
train_valYoloDetect = pd.read_csv('../user_data/featuresDetect/b_trainValFeatureYoloV51024.csv')
test_dataYoloDetect = pd.read_csv('../user_data/featuresDetect/b_testFeatureYoloV51024.csv')
# key_frame
train_valYoloDetect_key = train_valYoloDetect[train_valYoloDetect['name'].isin(train_key)].reset_index(drop=True)
train_valYoloDetect_key = process_detectYoloFrtKey(train_valYoloDetect_key, 'train')
test_dataYoloDetect_key = test_dataYoloDetect[test_dataYoloDetect['name'].isin(test_key)].reset_index(drop=True)
test_dataYoloDetect_key = process_detectYoloFrtKey(test_dataYoloDetect_key, 'test')
#
train_valYoloDetect = process_detectYoloFrt(train_valYoloDetect, 'train')
test_dataYoloDetect = process_detectYoloFrt(test_dataYoloDetect, 'test')
# prob feature from image-clssify based on xception
train_valCnn = pd.read_csv('../user_data/featuresDetect/trainValRoundB.csv')
test_dataCnn = pd.read_csv('../user_data/featuresDetect/testRoundB.csv')
train_valCnn = process_cnnFrt(train_valCnn)
test_dataCnn = process_cnnFrt(test_dataCnn)
# print(train_valAdd)
path = "../data/"  # 存放原始数据的地址
result_path = "../prediction_result/"  # 存放输出的地址
if not os.path.exists(result_path):
    os.makedirs(result_path)
train_json = pd.read_json("../user_data/amap_traffic_annotations_train.json")
test_json = pd.read_json(path + "amap_traffic_annotations_b_test_0828.json")
#
train_df = get_data(train_json[:])
test_df = get_data(test_json[:])

# merge features
train_df = train_df.merge(train_valCnn)
train_df = train_df.merge(train_valYoloDetect)
train_df = train_df.merge(train_valYoloDetect_key)
#
test_df = test_df.merge(test_dataCnn)
test_df = test_df.merge(test_dataYoloDetect)
test_df = test_df.merge(test_dataYoloDetect_key)

# features
detectYolofeatures = list(train_valYoloDetect.columns)
detectYolofeatureskey = list(train_valYoloDetect_key.columns)
cnnFeatures = ['p_0', 'p_1', 'p_2']
detectYolofeatures.remove('map_id')
detectYolofeatureskey.remove('map_id')
time_features = ["map_id", "month_mean", "hour_mean", "minute_mean"]
#
select_features = time_features + detectYolofeatures + detectYolofeatureskey + cnnFeatures

train_data = train_df[select_features].copy()
kind = train_df["label"]
test_data = test_df[select_features].copy()

train_data.drop(['map_id'], axis=1, inplace=True)
test_names = test_data['map_id'].values
test_data.drop(['map_id'], axis=1, inplace=True)

print('train_val:', train_data.shape, 'test:', test_data.shape)

#
#三个树模型加权五折交叉验证
xlf,llf,clf=models()
#
details = []
answers = []
mean_f1=0
n_splits=5
sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
for train, test in sk.split(train_data, kind):
    x_train = train_data.iloc[train]
    y_train = kind.iloc[train]
    x_test = train_data.iloc[test]
    y_test = kind.iloc[test]

    xlf.fit(x_train, y_train)
    pred_xgb = xlf.predict(x_test)
    weight_xgb = eval_score(y_test,pred_xgb)

    llf.fit(x_train, y_train)
    pred_llf = llf.predict(x_test)
    weight_lgb = eval_score(y_test,pred_llf)

    clf.fit(x_train, y_train)
    pred_cab = clf.predict(x_test)
    weight_cab =  eval_score(y_test,pred_cab)

    prob_xgb = xlf.predict_proba(x_test)
    prob_lgb = llf.predict_proba(x_test)
    prob_cab = clf.predict_proba(x_test)

    scores = []
    ijk = []
    weight = np.arange(0, 1.05, 0.1)
    for i, item1 in enumerate(weight):
        for j, item2 in enumerate(weight[weight <= (1 - item1)]):
            prob_end = prob_xgb * item1 + prob_lgb * item2 + prob_cab * (1 - item1 - item2)
            score = eval_score(y_test, np.argmax(prob_end, axis=1))
            scores.append(score)
            ijk.append((item1, item2, 1 - item1 - item2))

    ii = ijk[np.argmax(scores)][0]
    jj = ijk[np.argmax(scores)][1]
    kk = ijk[np.argmax(scores)][2]

    details.append(max(scores))
    details.append(weight_xgb)
    details.append(weight_lgb)
    details.append(weight_cab)
    details.append(ii)
    details.append(jj)
    details.append(kk)

    print(max(scores))
    mean_f1+=max(scores)/n_splits

    test_xgb = xlf.predict_proba(test_data)
    test_lgb = llf.predict_proba(test_data)
    test_cab = clf.predict_proba(test_data)
    ans = test_xgb * ii + test_lgb * jj + test_cab * kk

    answers.append(np.argmax(ans, axis=1))
print('mean weighted f1:',mean_f1)
#
fina=[]
for i in range(len(test_data)):
    counts=np.bincount(np.array(answers,dtype='int')[:,i])
    fina.append(np.argmax(counts))

#
def convertSeq(x):
    x=str(x)
    seq=''.join(['0' for i in range(6-len(x))])
    return seq+x
pres_dic={}
for i in range(len(fina)):
    pres_dic[convertSeq(test_names[i])]=fina[i]


import json
# some parameters
rawLabelDir='../data/amap_traffic_annotations_b_test_0828.json'
with open(rawLabelDir) as f:
    d=json.load(f)

cnt_statistic={'畅通':0,'缓行':0,'拥堵':0}
annos=d['annotations']
for i in range(len(annos)):
    anno=annos[i]
    imgId=anno['id']
    frame=anno['key_frame'][:-4]
    status=pres_dic[imgId]
    d['annotations'][i]['status']=int(status)
    if status==0:
        cnt_statistic['畅通']+=1
    elif status==1:
        cnt_statistic['缓行']+=1
    else:
        cnt_statistic['拥堵']+=1

print(cnt_statistic)
#
cls_0=cnt_statistic['畅通']
cls_1=cnt_statistic['缓行']
cls_2=cnt_statistic['拥堵']
json_data=json.dumps(d)
save_dir=result_path+'result.json'
with open(save_dir,'w') as w:
    w.write(json_data)

#
