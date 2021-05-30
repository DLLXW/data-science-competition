import pandas as pd
import numpy as np
import os
import lightgbm
from frt import *
from evaluation import uAUC,compute_weighted_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
ROOT_PATH="./data/"
FEED_EMBEDDING_DIR="data/wechat_algo_data1/feed_embeddings_PCA.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
#
import sys
class lgb_ctr(object):
    '''
    '''
    def __init__(self,stage,action):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'boost_from_average': True,
            'train_metric': True,
            'feature_fraction_seed': 1,
            'learning_rate': 0.05,
            'is_unbalance': True,  # 当训练数据是不平衡的，正负样本相差悬殊的时候，可以将这个属性设为true,此时会自动给少的样本赋予更高的权重
            'num_leaves': 128,  # 一般设为少于2^(max_depth)
            'max_depth': -1,  # 最大的树深，设为-1时表示不限制树的深度
            'min_child_samples': 15,  # 每个叶子结点最少包含的样本数量，用于正则化，避免过拟合
            'max_bin': 200,  # 设置连续特征或大量类型的离散特征的bins的数量
            'subsample': 1,  # Subsample ratio of the training instance.
            'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.5,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'subsample_for_bin': 200000,  # Number of samples for constructing bin
            'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            'reg_alpha': 2.99,  # L1 regularization term on weights
            'reg_lambda': 1.9,  # L2 regularization term on weights
            'nthread': 12,
            'verbose': 0,
            'force_row_wise':True
        }
        self.stage=stage
        self.action=action
        self.select_frts=[]
        self.select_frts+=SELECT_FRTS
        #feed embedding by PCA
        #self.select_frts+=['feed_embed_'+str(i) for i in range(32)]
    def process_data(self,train_path,test_path):
        df_train = pd.read_csv(train_path)
        print(df_train.columns)
        df_test=pd.read_csv(test_path)
        df=pd.concat((df_train,df_test)).reset_index(drop=True)
        for feature in ONE_HOT_FEATURE:
            df[feature] = LabelEncoder().fit_transform(df[feature].apply(str))
        #df_feed=pd.read_csv(feed_embedding_dir)
        #df=df.merge(df_feed)
        for col in ['userid','feedid','device','authorid','bgm_song_id','bgm_singer_id']:
            df[col] = LabelEncoder().fit_transform(df[col].apply(str))
        train=df.iloc[:df_train.shape[0]].reset_index(drop=True)
        test=df.iloc[df_train.shape[0]:].reset_index(drop=True)
        #
        return train,test
    def train_test(self):
        #读取训练集数据
        train_path = ROOT_PATH +f'/train_data_for_{self.action}.csv'
        test_path = ROOT_PATH + '/test_data.csv'
        df_train,df_test=self.process_data(train_path,test_path)
        #
        if self.stage=='offline_train':
            df_val=df_train[df_train['date_']==14].reset_index(drop=True)
            df_train=df_train[df_train['date_']<14].reset_index(drop=True)
        else:
            df_val=None
        #
        train_x=df_train[self.select_frts]
        train_y=df_train[self.action]
        print(train_x.shape,train_y.shape)
        train_matrix = lightgbm.Dataset(train_x, label=train_y)
         #-----------
        self.model=lightgbm.train(self.params, train_matrix
                    ,num_boost_round=200
                   )
        print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(list(train_x.columns), self.model.feature_importance("gain")),
                        key=lambda x: x[1],reverse=True))[:5]))
        if self.stage=='offline_train':
            return self.evaluate(df_val)
        elif self.stage=='online_train':
            return self.predict(df_test)
    def evaluate(self,df):
        #测试集
        test_x=df[self.select_frts].values
        labels=df[self.action].values
        userid_list = df['userid'].astype(str).tolist()
        logits = self.model.predict(test_x)
        uauc=uAUC(labels,logits,userid_list)
        return df[["userid","feedid"]],logits,uauc
    def predict(self,df):
        #测试集
        test_x=df[self.select_frts].values
        logits= self.model.predict(test_x)
        return df[["userid","feedid"]],logits
def main(argv):
    stage = argv[1]
    eval_dict = {}
    predict_dict = {}
    ids = None
    submit = pd.read_csv(ROOT_PATH + '/test_data.csv')[['userid', 'feedid']]
    for action in ACTION_LIST:
        print("-------------------Action-----------------:", action)
        model = lgb_ctr(stage,action)
        if stage =="offline_train":
            # 离线训练并评估
            ids,logits,action_uauc=model.train_test()
            eval_dict[action]=action_uauc
            predict_dict[action] = logits

        elif stage == "online_train":
            # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
            ids, logits = model.train_test()
            predict_dict[action] = logits

        else:
            print("stage must be in [online_train,offline_train]")
    #
    if stage =="offline_train":
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)

    if stage =="online_train":
        # 计算所有行为的加权uAUC
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], sort=False, axis=1)
        # 写文件
        file_name = "submit_lgb.csv"
        submit_file = os.path.join(ROOT_PATH, 'submit', file_name)
        print('Save to: %s'%submit_file)
        res.to_csv(submit_file, index=False)
if __name__ == "__main__":
    main(sys.argv)