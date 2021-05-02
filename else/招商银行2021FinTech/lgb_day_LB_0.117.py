import pandas as pd
import numpy as np
import os
import lightgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
plt.close("all")
#
#----------树模型直接预测每一天的流量，特征:year,month,day,weekday,WKD_TYP_CD-----
train_df=pd.read_csv('./data/train_v1.csv')
test_df=pd.read_csv('./data/test_v1_periods.csv')#按0.5h计算
test_day=pd.read_csv('./data/test_v1_day.csv')#按天计算
wkd_df=pd.read_csv('./data/wkd_v1.csv')
wkd_df=wkd_df.rename(columns={'ORIG_DT':'date'})
train_df=train_df.merge(wkd_df)
#处理特征
def get_frt(df):
    df['WKD_TYP_CD']=df['WKD_TYP_CD'].map({'WN':0,'SN': 1, 'NH': 1, 'SS': 1, 'WS': 0})
    df['date']=pd.to_datetime(df['date'])
    df['dayofweek']=df['date'].dt.dayofweek+1
    df['day']=df['date'].dt.day
    df['month']=df['date'].dt.month
    df['year']=df['date'].dt.year
    df.drop(['date','post_id'],axis=1,inplace=True)
    return df
#train
tmp=train_df[['date','post_id','amount']].groupby(['date','post_id'],sort=False).agg('sum')
train_day_df=pd.DataFrame(tmp).reset_index()
train_day_df_A=train_day_df[train_day_df['post_id']=='A'].reset_index(drop=True)
train_day_df_B=train_day_df[train_day_df['post_id']=='B'].reset_index(drop=True)
train_day_df_A=train_day_df_A.merge(wkd_df)
train_day_df_B=train_day_df_B.merge(wkd_df)
train_day_df_A=get_frt(train_day_df_A)
train_day_df_B=get_frt(train_day_df_B)
train_day_df_A['amount']=train_day_df_A['amount']/1e4
train_day_df_B['amount']=train_day_df_B['amount']/1e4
#test
tmp=test_df[['date','post_id']].groupby(['date','post_id'],sort=False).agg('sum')
test_day_df=pd.DataFrame(tmp).reset_index()
test_day_df_A=test_day_df[test_day_df['post_id']=='A'].reset_index(drop=True)
test_day_df_B=test_day_df[test_day_df['post_id']=='B'].reset_index(drop=True)
test_day_df_A=test_day_df_A.merge(wkd_df)
test_day_df_B=test_day_df_B.merge(wkd_df)
test_day_df_A=get_frt(test_day_df_A)
test_day_df_B=get_frt(test_day_df_B)
#训练集和测试集
print(train_day_df_A.shape,test_day_df_A.shape)#(1035, 6) (30, 5)
def lgb_cv(train_x, train_y, test_x):
    predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    folds = 10
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train = np.zeros((train_x.shape[0]))
    test = np.zeros((test_x.shape[0]))
    test_pre = np.zeros((folds, test_x.shape[0]))
    test_pre_all = np.zeros((folds, test_x.shape[0]))
    cv_scores = []
    tpr_scores = []
    cv_rounds = []

    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        train_matrix = lightgbm.Dataset(tr_x, label=tr_y)
        test_matrix = lightgbm.Dataset(te_x, label=te_y)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metrics':'mean_squared_error',
            'num_leaves': 2 ** 5-1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'learning_rate': 0.05,
            'seed': 2021,
            'nthread': 8,
            'verbose': -1,
        }
        num_round = 4000
        early_stopping_rounds = 100
        if test_matrix:
            model = lightgbm.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=200,
                              #feval=tpr_eval_score,
                              early_stopping_rounds=early_stopping_rounds
                              )
            print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                        key=lambda x: x[1],reverse=True))[:10]))
            importance_list=[ x[0] for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                        key=lambda x: x[1],reverse=True))]
            #print(importance_list)
            pre = model.predict(te_x, num_iteration=model.best_iteration)#
            pred = model.predict(test_x, num_iteration=model.best_iteration)#
            train[test_index] = pre
            test_pre[i, :] = pred
            cv_scores.append(mean_squared_error (te_y, pre))
            cv_rounds.append(model.best_iteration)
            test_pre_all[i, :] = pred
        #
        print("cv_score is:", cv_scores)
    use_mean=True
    if use_mean:
        test[:] = test_pre.mean(axis=0)
    else:
        pass
    #
    print("val_mean:" , np.mean(cv_scores))
    print("val_std:", np.std(cv_scores))
    return train, test, test_pre_all, np.mean(cv_scores),importance_list
if __name__=="__main__":
    select_frts=['WKD_TYP_CD','year','month','day','dayofweek']
    train_df=train_day_df_A#训练集A
    train_df=train_df[(train_df['year']==2020) & (train_df['month']>5)].reset_index(drop=True)
    test_df=test_day_df_A#测试集A
    train_x = train_df[select_frts].copy()
    train_y = train_df['amount']
    test_x = test_df[select_frts].copy()
    print(train_x.shape,train_y.shape,test_x.shape)
    lgb_train, lgb_test, sb, cv_scores, importance_list = lgb_cv(train_x, train_y, test_x)
    lgb_test_A=[item if item>0 else 0 for item in lgb_test]
    #
    train_df=train_day_df_B#训练集B
    train_df=train_df[(train_df['year']==2020) & (train_df['month']>5)].reset_index(drop=True)
    test_df=test_day_df_B#测试集B
    train_x = train_df[select_frts].copy()
    train_y = train_df['amount']
    test_x = test_df[select_frts].copy()
    print(train_x.shape,train_y.shape,test_x.shape)
    lgb_train, lgb_test, sb, cv_scores, importance_list = lgb_cv(train_x, train_y, test_x)
    lgb_test_B=[item if item>0 else 0 for item in lgb_test]
    print(np.mean(lgb_test_A),np.sum(lgb_test_A),np.mean(lgb_test_B),np.sum(lgb_test_B))
    #
    pre_A=np.array(lgb_test_A)
    pre_B=np.array(lgb_test_B)
    test_day=pd.read_csv('./data/test_v1_day.csv')#按天计算
    pre_day=[]
    for i in range(30):
        pre_day.append(pre_A[i]*1e4)
        pre_day.append(pre_B[i]*1e4)
    test_day['amount']=pre_day
    #
    if not os.path.exists('submitTree/'):
        os.makedirs('submitTree/')
    f=open('submitTree/test_day_day.txt','w')
    f.write('Date'+','+'Post_id'+','+'Predict_amount'+'\n')
    for _,date,post_id,amount in test_day.itertuples():
        f.write(date+','+post_id+','+str(int(amount))+'\n')
    f.close()