
#null特征筛选
from sklearn.model_selection import train_test_split
import random
def nullImportant(train_df,select_frts):
    train_data=train_df[select_frts]
    kind=train_df['label']
    x_train,x_val,y_train,y_val=train_test_split(train_data,kind,test_size=0.1,random_state=8)
    train_matrix = lightgbm.Dataset(x_train, label=y_train)
    test_matrix = lightgbm.Dataset(x_val, label=y_val)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        #'scale_pos_weight':20,
        'metrics':'auc',#'binary_logloss',
        'num_leaves': 2 ** 6-1,
        #'lambda_l2': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'learning_rate': 0.05,
        'seed': 2020,
        'nthread': 8,
        'num_class': 1,
        'verbose': -1,
    }
    num_round = 4000
    early_stopping_rounds = 500
    model = lightgbm.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=200,
                    #feval=tpr_eval_score,
                    early_stopping_rounds=early_stopping_rounds
                    )
    important_normal=sorted(zip(list(x_train.columns), model.feature_importance("gain")),key=lambda x: x[0],reverse=True)
    print('finished .......')
    #
    list_neg=[]
    len_round=10
    for round_i in range(len_round):
        train_data=train_df[select_frts]
        re_label=train_df["label"].values
        random.shuffle(re_label)
        train_df["label"]=re_label
        #
        kind=train_df['label']
        x_train,x_val,y_train,y_val=train_test_split(train_data,kind,test_size=0.1,random_state=8)
        train_matrix = lightgbm.Dataset(x_train, label=y_train)
        test_matrix = lightgbm.Dataset(x_val, label=y_val)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            #'scale_pos_weight':20,
            'metrics':'auc',#'binary_logloss',
            'num_leaves': 2 ** 6-1,
            #'lambda_l2': 10,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'learning_rate': 0.05,
            'seed': 2020,
            'nthread': 8,
            'num_class': 1,
            'verbose': -1,
        }
        num_round = 400
        early_stopping_rounds = 200
        model = lightgbm.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=200,
                        #feval=tpr_eval_score,
                        #early_stopping_rounds=early_stopping_rounds
                        )
        #
        list_neg.append(sorted(zip(list(x_train.columns), model.feature_importance("gain")),key=lambda x: x[0],reverse=True))
    #
    print('finished .......')
    important_no=[]
    vv=list_neg[0]
    for i in range(len(vv)):
        key=vv[i][0]
        value=vv[i][1]/len_round
        for j in range(1,len_round):
            value+=list_neg[j][i][1]/len_round
        important_no.append((key,value))
    #
    diff_list=[]
    diff_list_div=[]
    for i in range(len(important_normal)):
        diff_list.append((important_normal[i][0],important_normal[i][1]-important_no[i][1]))
        diff_list_div.append((important_normal[i][0],important_normal[i][1]/(1e-8+important_no[i][1])))
    importance_last=sorted(diff_list,key=lambda x: x[1],reverse=True)
    importance_last_div=sorted(diff_list_div,key=lambda x: x[1],reverse=True)
    retrun importance_last
