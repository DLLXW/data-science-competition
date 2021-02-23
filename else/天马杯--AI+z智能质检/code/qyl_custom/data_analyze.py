import pandas as pd
import numpy as np
'''
下面是进行数据分析的代码:
原始数据:Train.csv,TestA.csv
总共135656个会话，2615989条记录
A榜测试集总共50069个会话,1092486条记录
'''
#------------------统计原始数据信息－－－－－－－－－－
train_df=pd.read_csv('../../data/Train.csv')
test_df=pd.read_csv('../../data/Test_B.csv')
print(train_df.head(2))
print("训练集总共{}个会话,{}条记录".format(len(train_df['SessionId'].unique()),train_df.shape[0]))
print("B榜测试集总共{}个会话,{}条记录".format(len(test_df['SessionId'].unique()),test_df.shape[0]))
#统计每一条对话的长度,以及正负类平均对话条数
train_cnt_len=[len(i) for i in train_df['Text'].values]#统计每一条对话的长度
test_cnt_len=[len(i) for i in test_df['Text'].values]
print('训练集对话长度　min:{},max:{},mean:{}'.format(np.min(train_cnt_len),np.max(train_cnt_len),np.mean(train_cnt_len)))
print('测试集对话长度　min:{},max:{},mean:{}'.format(np.min(test_cnt_len),np.max(test_cnt_len),np.mean(test_cnt_len)))
#可以看到文本长度及其不均衡,对话长度min:1,max:386,mean:11.27.
#接下来统计长文本的条数占比大于[20,30,50,100,200,300]占比
atten=[20,30,50,100]
train_cnt_len=sorted(train_cnt_len,reverse=True)
test_cnt_len=sorted(test_cnt_len,reverse=True)
for point in atten:
    cnt_train = 0
    cnt_test=0
    for i in range(len(train_cnt_len)):
        if train_cnt_len[i]>point:
            cnt_train+=1
        else:
            break
   #
    for i in range(len(test_cnt_len)):
        if test_cnt_len[i]>point:
            cnt_test+=1
        else:
            break
    print("_________________________统计对话长度分布____________________________")
    print("长度大于{}的对话占比:训练集{:.3}%;:测试集{:.3}%".format(point,100*cnt_train/len(train_cnt_len),100*cnt_test/len(test_cnt_len)))
#
'''
训练集对话长度　min:1,max:386,mean:11.272667430941032
测试集对话长度　min:1,max:449,mean:10.866262817097885
长度大于20的对话占比:训练集16.9%;:测试集16.0%
长度大于30的对话占比:训练集7.36%;:测试集6.94%
长度大于50的对话占比:训练集1.28%;:测试集1.26%
长度大于100的对话占比:训练集0.0594%;:测试集0.066%
长度大于200的对话占比:训练集0.00248%;:测试集0.00256%
长度大于300的对话占比:训练集0.000153%;:测试集0.000549%
!!!大于50的对话只占1.x%由此可见:设定最大长度为50是足够了
'''
#---------------------------统计类别有关的特征-------------------------------------
for mode in ['train','test']:
    if mode=='train':
        grouped = train_df.groupby('SessionId')
        new_df = {'SessionId':[],'Text':[],'HighRiskFlag':[]}
        pos_group_len = []
        neg_group_len = []
        train_group_len=[]
    else:
        grouped = test_df.groupby('SessionId')
        new_df = {'SessionId': [], 'Text': []}
        test_group_len=[]

    for name, group in grouped:
        #
        tmp=group['Text'].values
        new_str=''.join(tmp)

        new_df['SessionId'].append(name)
        new_df['Text'].append(new_str)
        if mode=='train':
            new_df['HighRiskFlag'].append(group['HighRiskFlag'].values[0])
            flag=group['HighRiskFlag'].values[0]
            train_group_len.append(len(group))
            if flag==1:
                pos_group_len.append(len(group))
            else:
                neg_group_len.append(len(group))
        else:
            test_group_len.append(len(group))
    if mode=='train':
        new_df=pd.DataFrame(new_df)
        #
        flag_list=new_df['HighRiskFlag'].values
        text_list=new_df['Text'].values
        pos_len=[]
        neg_len=[]
        for i in range(len(flag_list)):
            flag=flag_list[i]
            text=text_list[i]
            if flag==1:
                pos_len.append(len(text))
            else:
                neg_len.append(len(text))
        print("_________________________训练集按照SessionGroup____________________________")
        print('训练集_pos对话总长度:min:{};max:{};平均有{}个字对话'.format(np.min(pos_len),np.max(pos_len),np.mean(pos_len)))
        print('训练集_neg对话总长度:min:{};max:{};平均有{}个字对话'.format(np.min(neg_len), np.max(neg_len), np.mean(neg_len)))
        print('训练集对话总条数:min:{};max:{};平均有{}条对话'.format(np.min(train_group_len), np.max(train_group_len), np.mean(train_group_len)))
        print('训练集_pos对话总条数:min:{};max:{};平均有{}条对话'.format(np.min(pos_group_len), np.max(pos_group_len), np.mean(pos_group_len)))
        print('训练集_neg对话总条数:min:{};max:{};平均有{}条对话'.format(np.min(neg_group_len), np.max(neg_group_len), np.mean(neg_group_len)))
    else:
        new_df = pd.DataFrame(new_df)
        #
        text_list = new_df['Text'].values
        test_len = []
        for i in range(len(text_list)):
            text = text_list[i]
            test_len.append(len(text))
        print("_________________________测试集按照SessionGroup____________________________")
        print('测试集对话总长度:min:{};max:{};平均有{}个字对话'.format(np.min(test_len), np.max(test_len), np.mean(test_len)))
        print('测试集对话总条数:min:{};max:{};平均有{}条对话'.format(np.min(test_group_len), np.max(test_group_len), np.mean(test_group_len)))

'''
_________________________训练集按照SessionGroup____________________________
训练集_pos对话总长度:min:1;max:16527;平均有1568.497487437186个字对话
训练集_neg对话总长度:min:1;max:16345;平均有205.38433163272896个字对话
训练集对话总条数:min:1;max:1317;平均有19.28399038745061条对话
训练集_pos对话总条数:min:1;max:1317;平均有125.3140703517588条对话
训练集_neg对话总条数:min:1;max:1210;平均有18.342461067067276条对话
_________________________测试集按照SessionGroup____________________________
测试集对话总长度:min:1;max:16247;平均有237.09760530467955个字对话
测试集对话总条数:min:1;max:1672;平均有21.819608939663265条对话
!!!!
    由此可以看到:对话的总长度和对话的总条数是一个区分正负类的非常重要的特征.需要想办法将该信息融入网络的训练过程中.
    这里可以衍生出一个想法，如果以单条记录训练模型，在预测过程中，本来我们往往利用的是对一个session的所有概率
    先进行从高到底排序，取前k条记录的平均，现在可以换一种方式，直接对每一种记录进行求和运算，这样就等于融入
'''
#-------------------------制作训练集（将多句合成一句）--------------------------------
print('-------------------------制作训练集（将多句合成一句）--------------------------------')

len_group=8
for mode in ['train','test']:
    print(mode)
    if mode=='train':
        grouped = train_df.groupby('SessionId',sort=False)
        new_df = {'SessionId':[],'Text':[],'HighRiskFlag':[]}
    else:
        grouped = test_df.groupby('SessionId',sort=False)
        new_df = {'SessionId': [], 'Text': []}
    for name, group in grouped:
        #
        tmp=group['Text'].values
        #
        if len(group)>len_group:
            for cut in range(len(group)//len_group):#将４句合成一句
                new_str=''.join(tmp[cut*len_group:(cut+1)*len_group])
                new_df['SessionId'].append(name)
                new_df['Text'].append(new_str)
                if mode == 'train':
                    new_df['HighRiskFlag'].append(group['HighRiskFlag'].values[0])
        else:
            new_str = ''.join(tmp)
            new_df['SessionId'].append(name)
            new_df['Text'].append(new_str)
            if mode == 'train':
                new_df['HighRiskFlag'].append(group['HighRiskFlag'].values[0])

    new_df=pd.DataFrame(new_df)
    if mode=='train':
        new_df.to_csv('../datasets/tianma_cup/train_group8_df.csv',index=False)
    else:
        new_df.to_csv('../datasets/tianma_cup/testB_group8_df.csv', index=False)
    print(new_df)