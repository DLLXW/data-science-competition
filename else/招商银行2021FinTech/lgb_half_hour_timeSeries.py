import pandas as pd
import numpy as np
import os
import lightgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
plt.close("all")
#
#----------树模型直接预测每一时间段的流量，特征:period,year,month,day,dayofweek,WKD_TYP_CD-----
#处理特征
# #train_hour_df_A
# 预测一天之类的各个时间段占一天的业务量
# '''
def get_frt(df):
    df['WKD_TYP_CD']=df['WKD_TYP_CD'].map({'WN':0,'SN': 1, 'NH': 1, 'SS': 1, 'WS': 0})
    df['date']=pd.to_datetime(df['date'])
    df['dayofweek']=df['date'].dt.dayofweek+1
    df['day']=df['date'].dt.day
    df['month']=df['date'].dt.month
    df['year']=df['date'].dt.year
    df.drop(['date','post_id'],axis=1,inplace=True)
    return df
#
def process_data():
    #
    #-----------------------数据处理-------------------------
    #
    train_df=pd.read_csv('./data/train_v1.csv')
    wkd_df=pd.read_csv('./data/wkd_v1.csv')
    wkd_df=wkd_df.rename(columns={'ORIG_DT':'date'})
    train_df=train_df.merge(wkd_df)
    #
    train_df['amount']=train_df['amount']/1e4
    train_hour_df_A_cls=train_df[train_df['post_id']=='A'].reset_index(drop=True)
    train_hour_df_B=train_df[train_df['post_id']=='B'].reset_index(drop=True)
    #
    name_A=['A'+str(i) for i in range(2,14)]
    train_hour_df_A=train_hour_df_A_cls[train_hour_df_A_cls['biz_type']=='A1'].reset_index(drop=True)
    new_train_amount=train_hour_df_A['amount'].values
    for Ai in name_A:
        tmp=train_hour_df_A_cls[train_hour_df_A_cls['biz_type']==Ai].reset_index(drop=True)
        new_train_amount+=tmp['amount'].values
    #
    train_hour_df_A['amount']=new_train_amount
    train_hour_df_A.drop(['biz_type'],axis=1,inplace=True)
    train_hour_df_B.drop(['biz_type'],axis=1,inplace=True)
    train_hour_df_A=get_frt(train_hour_df_A)
    train_hour_df_B=get_frt(train_hour_df_B)
    #-----------
    test_df=pd.read_csv('./data/test_v1_periods.csv')#按0.5h计算
    test_day=pd.read_csv('./data/test_v1_day.csv')#按天计算
    test_hour_df_A=test_df[test_df['post_id']=='A'].reset_index(drop=True)
    test_hour_df_B=test_df[test_df['post_id']=='B'].reset_index(drop=True)
    test_hour_df_A=test_hour_df_A.merge(wkd_df)
    test_hour_df_B=test_hour_df_B.merge(wkd_df)
    test_hour_df_A=get_frt(test_hour_df_A)
    test_hour_df_B=get_frt(test_hour_df_B)
    test_hour_df_A.drop(['amount'],axis=1,inplace=True)
    test_hour_df_B.drop(['amount'],axis=1,inplace=True)
    #
    print(train_hour_df_A.shape,test_hour_df_A.shape)#(1035, 7) (30, 6)
    #------------------------数据处理完毕-------------------------
    return train_hour_df_A,train_hour_df_B,test_hour_df_A,test_hour_df_B
class lgb_time_series(object):
    '''
    lgb:用于时序预测，滚动方式，前一个时间点的预测结果用于构建下一个时间点的特征
    注意:
    1.利用滑动窗口的方式构建训练集
    2.lgb模型只需要用第一步的训练集一次
    3.首先预测测试集的第一个值
    4.将测试集的第一个值加入滑动窗口，用于构建第二个值的特征
    5.重复4,直到所有测试样本预测完毕

    重点在于测试集特征的滚动构建！！！
    '''
    def __init__(self,train_df,test_df,select_frts,label_col='amount',slide_window=5):
        self.params = {
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
        self.predict_period=len(test_df)#预测时长
        self.slide_window=slide_window#用来构建训练样本的滑动窗口长度
        shift_cols=['shift_'+str(i) for i in range(1,self.slide_window)]
        ratio_cols=['ratio_'+str(i) for i in range(1,self.slide_window-1)]
        self.select_frts=select_frts+shift_cols+ratio_cols
        self.label_col=label_col
        self.pre_cnt=0
        self.train_df=train_df
        self.test_df=test_df
        self.test_df[self.label_col]=np.NaN#测试集标签用nan填充
        self.model=None
    def make_train_dataset(self,train_dataset):
        #滑动窗口构建训练集
        shift_cols=[]
        for s in range(1,self.slide_window):
            shift_cols.append('shift_'+str(s))
            train_dataset['shift_'+str(s)]=train_dataset[self.label_col].shift(s)#shift操作
        #根据滑动窗口内的数值构建新特征
        train_dataset=train_dataset[self.slide_window:].reset_index(drop=True)
        ratio_df=(train_dataset[shift_cols].shift(1,axis=1)-train_dataset[shift_cols])/(train_dataset[shift_cols]+1e-6)
        ratio_df=ratio_df.reset_index(drop=True)
        ratio_df.drop(['shift_1'],axis=1,inplace=True)
        ratio_cols=['ratio_'+str(i) for i in range(1,self.slide_window-1)]
        ratio_df.columns=ratio_cols
        train_dataset=pd.concat((train_dataset,ratio_df),axis=1)
        #print(train_dataset)
        return train_dataset #该train_dataset包含所有训练集以及第一条测试集
    #
    def fit_model(self,train_dataset):
        #用滑动窗口构建的训练集拟合模型,并且得到测试集第一个样本的预测结果
        train_x=train_dataset[self.select_frts]
        train_y=train_dataset[self.label_col]
        test_x=train_dataset[-1:].reset_index(drop=True)#获取测试集第一个样本
        test_x.drop([self.label_col],axis=1,inplace=True)
        #
        train_matrix = lightgbm.Dataset(train_x, label=train_y)
        test_x = test_x.values
        #
        self.model=lightgbm.train(self.params, train_matrix, num_boost_round=100)
        print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(list(train_x.columns), self.model.feature_importance("gain")),
                        key=lambda x: x[1],reverse=True))[:5]))
        first_pred = self.model.predict(test_x)#test_matrix
        print('first_pred:',first_pred[0])
        return first_pred[0]
    def predict(self,pre_pred):
        '''
        pre_pred:已经得到的前一个预测结果
        这里要将原始已经得到的预测结果逐个加入原始测试集
        '''
        self.test_df.loc[self.pre_cnt-1,self.label_col]=pre_pred#将预测结果填入测试集
        #测试集的前self.pre_cnt个样本接到训练集之后
        #但为了节约时间和内存，这里只需要：pd.concat((self.train_df[-self.slide_window:],self.test_df[:self.pre_cnt+1]))即可
        train_dataset=pd.concat((self.train_df[-self.slide_window:],self.test_df[:self.pre_cnt+1])).reset_index(drop=True)#测试集的前self.pre_cnt个样本接到训练集之后
        train_dataset=self.make_train_dataset(train_dataset)#滑动窗口构建训练集
        test_x=train_dataset[-1:].reset_index(drop=True)#获取测试集第self.pre_cnt个样本
        test_x.drop([self.label_col],axis=1,inplace=True)
        #预测阶段的滚动预测模型
        #print(test_x)
        test_x = test_x.values
        pred = self.model.predict(test_x)#test_matrix
        #print('cur pred:',pred)
        return pred
    def fit(self):
        train_dataset=pd.concat((self.train_df,self.test_df[:1])).reset_index(drop=True)#测试集的第一个样本接到训练集之后
        train_dataset=self.make_train_dataset(train_dataset)#滑动窗口构建训练集
        first_pred=self.fit_model(train_dataset)#训练并预测第一个结果
        pre_pred=first_pred
        for i in range(self.predict_period):
            #滚动预测
            self.pre_cnt+=1
            cur_pred=self.predict(pre_pred)
            pre_pred=cur_pred
            #
        return self.test_df

if __name__=="__main__":
    train_hour_df_A,train_hour_df_B,test_hour_df_A,test_hour_df_B=process_data()
    select_frts=['periods','WKD_TYP_CD','year','month','day','dayofweek']
    slide_window=5
    #
    train_df=train_hour_df_A#训练集A
    train_df=train_df[(train_df['year']==2020) & (train_df['month']>9)].reset_index(drop=True)
    #train_df=train_df[train_df['year']>2019].reset_index(drop=True)
    test_df=test_hour_df_A#测试集A
    # print(train_df)
    # print(test_df)
    lgb_time_A=lgb_time_series(train_df,test_df,select_frts=select_frts,slide_window=slide_window)
    lgb_test=lgb_time_A.fit()
    lgb_test=lgb_test['amount']
    lgb_test_A=[item if item>0 else 0 for item in lgb_test]
    #print(np.mean(lgb_test_A),np.sum(lgb_test_A))
    del lgb_time_A
    #
    train_df=train_hour_df_B#训练集B
    train_df=train_df[(train_df['year']==2020) & (train_df['month']>9)].reset_index(drop=True)
    #train_df=train_df[train_df['year']>2019].reset_index(drop=True)
    test_df=test_hour_df_B#测试集B
    lgb_time_B=lgb_time_series(train_df,test_df,select_frts=select_frts,slide_window=slide_window)
    lgb_test=lgb_time_B.fit()
    lgb_test=lgb_test['amount']
    lgb_test_B=[item if item>0 else 0 for item in lgb_test]
    del lgb_time_B
    print(np.mean(lgb_test_A),np.sum(lgb_test_A),np.mean(lgb_test_B),np.sum(lgb_test_B))
    # #
    test_df=pd.read_csv('./data/test_v1_periods.csv')#按0.5h计算
    pre_period=[]
    pre_hour_A=lgb_test_A
    pre_hour_B=lgb_test_B
    for i in range(30):#每一天
        for j in range(48):#每一个时间段
            pre_period.append(1e4*pre_hour_A[48*i+j])
        for j in range(48):
            pre_period.append(1e4*pre_hour_B[48*i+j])
    test_df['amount']=pre_period
    #
    save_dir='submitTreeSeries/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f=open(save_dir+'/test_df_day.txt','w')
    f.write('Date'+','+'Post_id'+','+'Periods'+','+'Predict_amount'+'\n')
    for _,date,post_id,periods,amount in test_df.itertuples():
        f.write(date+','+post_id+','+str(periods)+','+str(int(amount))+'\n')
    f.close()
    #