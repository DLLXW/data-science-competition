# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from frt import *
from deep_model import MyDeepFM
# 存储数据的根目录
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
# 
NUM_EPOCH_DICT = {"read_comment": 5, "like": 5, "click_avatar": 5,"forward": 5,
                                "comment": 1, "follow": 1, "favorite": 1, }

def main():
    submit = pd.read_csv(ROOT_PATH + '/test_data.csv')[['userid', 'feedid']]
    for action in ACTION_LIST:
        print('-----------action-----------',action)
        USE_FEAT = [action] + SELECT_FRTS
        train = pd.read_csv(ROOT_PATH +f'/train_data_for_{action}.csv')[USE_FEAT]
        #train = train.sample(frac=1., random_state=42).reset_index(drop=True)
        print("posi prop:")
        print(sum((train[action]==1)*1)/train.shape[0])
        test = pd.read_csv(ROOT_PATH + '/test_data.csv')[SELECT_FRTS]
        target = [action]
        test[target[0]] = 0
        test = test[USE_FEAT]
        data = pd.concat((train, test)).reset_index(drop=True)
        print(train.shape,test.shape,data.shape)
        dense_features = DENSE_FEATURE
        sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target]

        data[sparse_features] = data[sparse_features].fillna(0)
        data[dense_features] = data[dense_features].fillna(0)

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        # 2.count #unique features for each sparse field,and record dense feature field name
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                            for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]
        #
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                   for feat in sparse_features]
        #linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(drop=True)
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}
        #-------
        eval_ratio=0.
        eval_df=train.iloc[int((1-eval_ratio)*train.shape[0]):].reset_index(drop=True)
        userid_list=eval_df['userid'].astype(str).tolist()
        print('val len:',len(userid_list))

        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        model = MyDeepFM(linear_feature_columns=linear_feature_columns,
                        dnn_feature_columns=dnn_feature_columns,
                        use_fm=True,
                        dnn_hidden_units=(512,256),
                        l2_reg_linear=1e-1, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                        dnn_dropout=0.,
                        dnn_activation='relu', 
                        dnn_use_bn=False, task='binary', device=device)

        model.compile("adagrad", "binary_crossentropy", metrics=["auc"])

        history = model.fit(train_model_input, train[target].values, batch_size=1024,
         epochs=NUM_EPOCH_DICT[action], verbose=1,
                            validation_split=eval_ratio,userid_list=userid_list)
        pred_ans = model.predict(test_model_input, 128)
        submit[action] = pred_ans
        torch.cuda.empty_cache()
    # 保存提交文件
    submit.to_csv("./submit_base_deepfm.csv", index=False)
if __name__ == "__main__":
    main()
