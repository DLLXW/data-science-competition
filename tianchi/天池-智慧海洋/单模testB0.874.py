import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

print("导入依赖库(pandas,numpy,sklearn)成功..........")
#feature extract
#下面这个是进行特征抽取的类
class get_features():
    def __init__(self,path,train):
        self.train=train
        self.features=[]
        self.path=path
    #训练和测试数据分别进行处理
    def results(self):
        if self.train: #提取训练数据
            for i in range(7000):
                path=self.path+str(i)+".csv"
                self.build(path)
        else:         #提取测试数据
            for i in range(9000,11000):
                path=self.path+str(i)+".csv"
                self.build(path)
    def build(self,path):
        df=pd.read_csv(path)
        
        k=df['y']/df['x']
        self.features.append(k.min())#k_min
        self.features.append(k.max())#k_max
        self.features.append(k.mean())#k_mean
        
        #b=df['y']-k.mean()*df['x']
        #self.features.append(b.mean())#b_mean
        
        self.features.append(df['x'].max())#x_max
        #self.features.append(df['x'].mean())#x_mean
        self.features.append(df['x'].quantile(0.25))#x_1/4
        self.features.append(df['x'].quantile(0.5))#x_1/2
        #self.features.append(df['x'].quantile(0.75))#x_3/4

        self.features.append(df['y'].max())#y_max
        self.features.append(df['y'].mean())#y_mean
        self.features.append(df['y'].quantile(0.25))#y_1/4
        #self.features.append(df['y'].quantile(0.5))#y_1/2
        self.features.append(df['y'].quantile(0.75))#y_3/4
        
        #self.features.append(df['x'].cov(df['y']))#xy_cov
        
        #area=np.sqrt((df['x'].quantile(0.9)-df['x'].quantile(0.1))*(df['y'].quantile(0.9)-df['y'].quantile(0.1)))
        #self.features.append(area) #area

        df['time']=pd.to_datetime(df['time'],format='%m%d %H:%M:%S')
        
        t_diff=df['time'].diff().iloc[1:].dt.total_seconds()
        x_diff=df['x'].diff().iloc[1:].abs()
        y_diff=df['y'].diff().iloc[1:].abs()
        dis=np.sqrt(sum(np.sqrt(x_diff**2+y_diff**2)))
        x_a_mean=(x_diff/t_diff).mean()
        y_a_mean=(y_diff/t_diff).mean()

        #self.features.append(dis)#dis
        #self.features.append(np.sqrt(x_a_mean**2+y_a_mean**2)) #a

        #self.features.append(df['速度'].mean())#v_mean
        self.features.append(df['速度'].std())#v_std
        self.features.append(df['速度'].quantile(0.75))#v_3/4
        
        v_diff=df['速度'].diff().iloc[1:].abs()
        self.features.append((v_diff/t_diff).mean())#a_mean
        self.features.append((v_diff/t_diff).std())#a_std


        #self.features.append(df['方向'].quantile(0.25))#d_1/4
        self.features.append(df['方向'].quantile(0.75))#d_3/4
        #self.features.append(df['方向'].std())#d_std
        d_diff=df['方向'].diff().iloc[1:].abs()
        self.features.append((d_diff/t_diff).mean())#d_dif_mean
        #self.features.append((d_diff/t_diff).std())#d_dif_std

        self.features.append(df['x'].skew())#x_skew
        self.features.append(df['x'].kurt())#x_kurt
        self.features.append(df['y'].skew())#y_skew
        self.features.append(df['y'].kurt())#y_kurt
        #self.features.append(df['速度'].skew())#v_skew
        self.features.append(df['速度'].kurt())#v_kurt
        self.features.append(df['方向'].skew())#d_skew
        #self.features.append(df['方向'].kurt())#d_kurt
        
        df['hour'] = df['time'].dt.hour
        #self.features.append(df['hour'].skew())#h_skew
        self.features.append(df['hour'].kurt())#h_kurt
        
        self.features.append(df['y'].max()-df['x'].min())#y_x
        self.features.append(df['x'].max()-df['y'].min())#x_y
        #self.features.append(df['y'].max()-df['y'].min()/(1+(df['x'].max()-df['x'].min())))#x_y_k
        
        #训练数据和测试数据在这里有区别，一个有标签，一个无标签                 
        if(self.train):
            if df["type"].iloc[0]=='拖网':
                self.features.append(2)
            elif df["type"].iloc[0]=='刺网':
                self.features.append(1)
            else:
                self.features.append(0)


#调用上面的get_features类来处理训练数据
print("开始处理训练数据，这需要花费几分钟时间..........")
train_features=[]
path_train="./data/hy_round1_train_20200102/"
feature_class=get_features(path_train,train=True)
feature_class.results()
train_features=feature_class.features

#train_data
train_data=pd.DataFrame(np.array(train_features).reshape(7000,int(len(train_features)/7000)))
train_data.columns=['k_min','k_max','k_mean',
                    'x_max','x_1/4','x_1/2',
                    'y_max','y_mean','y_1/4',
                    'y_3/4','v_std','v_3/4','a_mean','a_std','d_3/4',
                    'd_dif_mean','x_skew','x_kurt','y_skew','y_kurt',
                    'v_kurt','d_skew','h_kurt','y_x','x_y','type']

lables=train_data.type
train_data.drop(['type'],axis=1,inplace=True)
print("训练数据处理完成.........")
#调用上面的get_features类来处理测试数据
test_features=[]
path_test="./data/hy_round1_testB_20200221/"
feature_class=get_features(path_test,train=False)
feature_class.results()
test_features=feature_class.features

#test_data
print("开始处理测试数据..........")
test_data=pd.DataFrame(np.array(test_features).reshape(2000,int(len(test_features)/2000)))
test_data.columns=['k_min','k_max','k_mean',
                    'x_max','x_1/4','x_1/2',
                    'y_max','y_mean','y_1/4',
                    'y_3/4','v_std','v_3/4','a_mean','a_std','d_3/4',
                    'd_dif_mean','x_skew','x_kurt','y_skew','y_kurt',
                    'v_kurt','d_skew','h_kurt','y_x','x_y']
print('测试数据处理完成.........')
#利用交叉验证，训练随机森林模型，采取10折cv，最终的预测结果通过投票得出
rlf=RandomForestClassifier(n_estimators=100, oob_score=True) #随机森林模型

print("开始进行交叉验证...........")
ans=[]#存储10折cv的答案，用于最终投票
sk=StratifiedKFold(n_splits=10,shuffle=True,random_state=2020)
for train,test in sk.split(train_data,lables):
    x_train=train_data.iloc[train]
    y_train=lables.iloc[train]
    x_test=train_data.iloc[test]
    y_test=lables.iloc[test]
    
    rlf.fit(x_train,y_train)
    
    rlf_pre=rlf.predict(test_data)
    ans.append(rlf_pre)
    #print(f1_score(y_test,rlf.predict(x_test),average='macro'))#打印每一折的验证准确率
print("交叉验证完成.............")

#投票得出最终结果
print("开始对十折交叉验证的结果进行投票..........")
prediction=[]
for i in range(2000):
    prediction.append(np.argmax(np.bincount(np.array(ans,dtype='int')[:,i])))
results=pd.DataFrame(np.arange(9000,11000,1))
results["type"]=pd.Series(prediction).map({0:'围网',1:'刺网',2:'拖网'})
results.to_csv('results.csv',index=None, header=None,encoding="utf-8")
print("最终预测结果已经被存入results.csv文件........")
print("查看预测结果的前几行:")
print(results.head())

