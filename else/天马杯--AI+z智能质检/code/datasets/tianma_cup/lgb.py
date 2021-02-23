import lightgbm as lgb
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
# 根据模型的数量加载特征，特征以 model-i 命名，与run_classifier_cv.py 脚本中命名一致
# data_path 是UER-py训练集路径

# 指定提取出特征的数量
model_num = 1

def load_features(feature_path, data_path):

    features = []
    for i in range(0,model_num):
        train_x = np.load(feature_path + 'model-'+str(i)+'.npy')
        features.append(train_x)
    features = np.concatenate(features, axis=-1)

    if data_path:
        labels = []
        with open(data_path) as f:
            lines = f.readlines()
        for line in lines[1:]:
            labels.append(int(line.split('\t')[0])) # 这里假设在label在第一列

        return features, labels
    else:
        return features

# 加载训练集、测试集特征
train_data, train_labels = load_features('features/train/', 'train_group8.tsv')
print(train_data.shape)
test_data = load_features('features/testB/', None)
print(train_data.shape,test_data.shape)
# 设置 LGB 超参数,可以根据贝叶斯优化结果调整
params = {
    'task': 'train',
    'boosting_type': 'dart',  # 设置提升类型
    'objective': 'multiclass',
    'num_class': 2,
    'metric': 'multi_error',

    'save_binary': True,
    'max_bin': 63,
    'bagging_fraction': 0.4,
    'bagging_freq': 5,

    'feature_fraction': 0.1239,
    'lambda_l1':  5.364,
    'lambda_l2': 4.799   ,
    'learning_rate': 0.02089 ,
    'max_depth': 17 ,
    'min_data_in_leaf': 24,
    'min_gain_to_split':0.9906 ,
    'min_sum_hessian_in_leaf':1,
    'num_leaves': 57,
    'verbose' : -1
}
#featur... | lambda_l1 | lambda_l2 | learni... | max_depth | min_da... | min_ga... | min_su... | num_le... |
#   0.1239   |  5.364    |  4.799    |  0.02089  |  17.01    |  24.22    |  0.8594   |  0.9906   |  57.93
# 十折训练并保存模型

#指定模型名称以及保存在 models/ 目录下，一共会生成10个模型
model_name = 'ensemble_modelsB'
fold_num = 5

labels=train_labels
train_x = train_data
test_x = test_data

per_flod_num = len(labels) // 5+1

pred = np.zeros([test_data.shape[0],2])
for fold in range(fold_num):
    x_train = np.concatenate((train_x[0:fold * per_flod_num], train_x[(fold+1)*per_flod_num:]),axis = 0)
    x_val = train_x[fold * per_flod_num: (fold+1)*per_flod_num]
    y_train = labels[0:fold * per_flod_num] + labels[(fold+1)*per_flod_num:]
    y_val = labels[fold * per_flod_num: (fold+1)*per_flod_num]


    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

    gbm = lgb.train(params,lgb_train,valid_sets=lgb_eval, verbose_eval=0)

    val_pred = gbm.predict(x_val)

    val_pred = np.argmax(val_pred,axis=1)

    correct = np.sum(val_pred == y_val)

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(val_pred), correct, len(val_pred)))
    gbm.save_model(model_name+'/fold'+str(fold)+'.lgb')
    pred += gbm.predict(test_x)/5
#
result=pred[:,1]
test_csv=pd.read_csv('/home/admins/qyl/tianma/data/testB_group8_df.csv')
test_csv['Probability']=result
sub_df=test_csv[['SessionId','Probability']]
##以一段session里面最大的概率为准，groupby之后取概率最大
topk = 3  # 取top3是合理的,取top10会掉分严重
#
sub = {'SessionId': [], 'Probability': []}
#
grouped = sub_df.groupby('SessionId', sort=False)
for name, group in grouped:
    pro_arr = group['Probability'].values
    res = np.mean(sorted(pro_arr, reverse=True)[:topk])  # topK取平均
    sub['SessionId'].append(name)
    sub['Probability'].append(res)
#
sub = pd.DataFrame(sub)
sub.to_csv("YYX_ROUND_B.csv",index=False)
print(sub)