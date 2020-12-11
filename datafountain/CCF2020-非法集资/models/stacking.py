#
import xgboost
import lightgbm
import catboost
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)

from utils import *
import numpy as np
import pandas as pd
def cnt_result(xx):
    cnt_re = {0: 0, 1: 0}
    for a in xx:
        if a <= 0.5:
            cnt_re[0] += 1
        else:
            cnt_re[1] += 1
    return cnt_re
def k_fold_serachParmaters(model, train_val_data,
                           train_val_kind):
    mean_f1 = {'F': 0, 'valid_f1': 0, 'valid_p': 0, 'valid_r': 0}
    train_mean_f1 = {'F': 0, 'valid_f1': 0, 'valid_p': 0, 'valid_r': 0}
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    for train, test in sk.split(train_val_data, train_val_kind):
        x_train = train_val_data.iloc[train]
        y_train = train_val_kind.iloc[train]
        x_test = train_val_data.iloc[test]
        y_test = train_val_kind.iloc[test]

        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        val_score = eval_score(y_test, pred)
        train_pre = model.predict(x_train)
        train_score = eval_score(y_train, train_pre)

        mean_f1['F'] += val_score['F'] / n_splits
        mean_f1['valid_f1'] += val_score['valid_f1'] / n_splits
        mean_f1['valid_p'] += val_score['valid_p'] / n_splits
        mean_f1['valid_r'] += val_score['valid_r'] / n_splits
        train_mean_f1['F'] += train_score['F'] / n_splits
        train_mean_f1['valid_f1'] += train_score['valid_f1'] / n_splits
        train_mean_f1['valid_p'] += train_score['valid_p'] / n_splits
        train_mean_f1['valid_r'] += train_score['valid_r'] / n_splits

    return mean_f1['valid_f1'],train_mean_f1['valid_f1']

def stage_1_model(train_data, kind):
    lgb_params = {'num_leaves': 8
        , 'max_depth': 6
        , 'learning_rate': 0.05
        , 'n_estimators': 50
        , 'n_jobs': 8
                  }
    xgb_params = {'max_depth': 5
        , 'learning_rate': 0.05
        , 'n_estimators': 60
        ,'colsample_bytree':0.7
        ,'min_child_weight':5
        , 'n_jobs': 8
        }
    cab_params = {'iterations': 50
        , 'learning_rate': 0.05
        , 'depth': 6
        ,'l2_leaf_reg':6
        , 'silent': True
        , 'thread_count': 8
        , 'task_type': 'CPU'
                  # ,'cat_features':CAT_FEATURES_INT
                  }
    lgb = lightgbm.LGBMClassifier(**lgb_params)
    print('LGBMClassifier:', k_fold_serachParmaters(lgb, train_data, kind))
    xgb = xgboost.XGBClassifier(**xgb_params)
    print('XGBClassifier:', k_fold_serachParmaters(xgb, train_data, kind))
    cab = catboost.CatBoostClassifier(**cab_params)
    print('CatBoostClassifier:', k_fold_serachParmaters(cab, train_data, kind))

    return lgb_params,xgb_params,cab_params


def stage_2_model(x_train,y_train):
    lgb_params_stage2 = {'num_leaves': 6
        , 'max_depth': 6
        , 'learning_rate': 0.05
        , 'n_estimators': 80
        , 'n_jobs': 8
                         }
    xgb_params_stage2 = {'max_depth': 5
        , 'learning_rate': 0.05
        , 'n_estimators': 50
        ,'colsample_bytree':0.6
        ,'min_child_weight':10
        , 'n_jobs': 8
            }
    cab_params_stage2 = {'iterations': 55
        , 'learning_rate': 0.04
        , 'depth': 5
        , 'silent': True
        , 'thread_count': 8
        , 'task_type': 'CPU'
                         }

    #
    lgb_stage2 = lightgbm.LGBMClassifier(**lgb_params_stage2)
    print('LGBMClassifier:', k_fold_serachParmaters(lgb_stage2, pd.DataFrame(x_train), pd.DataFrame(y_train)))
    xgb_stage2 = xgboost.XGBClassifier(**xgb_params_stage2)
    print('XGBClassifier:', k_fold_serachParmaters(xgb_stage2, pd.DataFrame(x_train), pd.DataFrame(y_train)))
    cab_stage2 = catboost.CatBoostClassifier(**cab_params_stage2)
    print('CatBoostClassifier:', k_fold_serachParmaters(cab_stage2, pd.DataFrame(x_train), pd.DataFrame(y_train)))
    return lgb_params_stage2,xgb_params_stage2,cab_params_stage2

if __name__=="__main__":
    train_df = pd.read_csv("../usr_data/train_data.csv")
    kind=train_df['label']
    test_df = pd.read_csv("../usr_data/test_data.csv")
    # 特征筛选
    select_features, _ = select_frt_names(train_df, thres=0.1)
    train_data=train_df[select_features].fillna(-1)
    test_data = test_df[select_features].fillna(-1)
    #
    ntrain = train_data.shape[0]
    ntest = test_data.shape[0]
    SEED = 2020  # for reproducibility
    NFOLDS = 5  # set folds for out-of-fold prediction
    '''
    第一层的基模型
    '''
    #
    print("-------->>>>>第一层的基模型<<<<<--------")
    lgb_params,xgb_params,cab_params=stage_1_model(train_data, kind)

    lgb = SklearnHelper(clf=lightgbm.LGBMClassifier, seed=SEED, params=lgb_params)
    xgb = SklearnHelper(clf=xgboost.XGBClassifier, seed=SEED, params=xgb_params)
    cab = SklearnHelper(clf=catboost.CatBoostClassifier, seed=SEED, params=cab_params)
    #
    # #
    # #
    y_train = kind

    x_train = train_data.values  # Creates an array of the train data
    x_test = test_data.values  # Creats an array of the test data
    #
    print("-------->>>>>训练、测试数据<<<<<--------")
    print("y_train:{};train_data:{};test_data:{}".format(y_train.shape, train_data.shape, test_data.shape))
    #
    # Create our OOF train and test predictions. These base results will be used as new features
    print("-------->>>>>第一阶段训练oof<<<<<--------")
    lgb_oof_train, lgb_oof_test = get_oof(lgb, x_train, y_train, x_test, ntrain,ntest,NFOLDS)  # LGBClassifier
    print("LGBClassifier.............")
    xgb_oof_train, xgb_oof_test = get_oof(xgb, x_train, y_train, x_test, ntrain,ntest,NFOLDS)  # XGBClassifier
    print("XGBClassifier.............")
    cab_oof_train, cab_oof_test = get_oof(cab, x_train, y_train, x_test, ntrain,ntest,NFOLDS)  # CatClassifier
    print("CatClassifier.............")

    x_train = np.concatenate((lgb_oof_train, xgb_oof_train, cab_oof_train),
                             axis=1)
    x_test = np.concatenate((lgb_oof_test, xgb_oof_test, cab_oof_test), axis=1)
    print(x_train.shape,x_test.shape)
    print("Training Stage_1 is complete")
    lgb_params_stage2, xgb_params_stage2, cab_params_stage2=stage_2_model(x_train,y_train)

    print("Training is complete")
    print("------第二阶段训练开始--------")



    #
    predictions_stage2 = []
    use_vote = True
    votes = []
    lgb_stage2 = lightgbm.LGBMClassifier(**lgb_params_stage2)
    xgb_stage2 = xgboost.XGBClassifier(**xgb_params_stage2)
    cab_stage2 = catboost.CatBoostClassifier(**cab_params_stage2)
    for model_two_stage in [lgb_stage2, xgb_stage2, cab_stage2]:
        #
        model_two_stage.fit(x_train, y_train)
        predictions = model_two_stage.predict_proba(x_test)[:, 1]
        predictions_stage2.append(predictions)
        cnt_re = cnt_result(predictions)
        print("预测结果中合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    #
    predictions = sum(predictions_stage2) / len(predictions_stage2)  #
    if use_vote:
        votes = [(pre > 0.5).astype(int) for pre in predictions_stage2]

    # predictions=np.sqrt(sum(np.array(np.array(predictions_stage2))**2)/len(predictions_stage2))#平方平均
    # predictions=pow(np.prod(np.array(predictions_stage2), axis=0),1/len(predictions_stage2))#几何平均
    cnt_re = cnt_result(predictions)
    print("预测结果中合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    #
    votes = [(pre > 0.5).astype(int) for pre in predictions_stage2]
    vote_most = []
    for i in range(len(predictions_stage2[0])):
        vote_list = np.array(votes)[:, i]
        tmp = {0: 0, 1: 0}
        for k in vote_list:
            tmp[k] += 1
        #
        most = sorted(tmp.items(), key=lambda item: item[1])[-1][0]
        vote_most.append(most)
    cnt_re = cnt_result(vote_most)
    print("投票结果中合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    sub_df = test_df[['id']].copy()
    sub_df['score'] = vote_most
    sub_df.to_csv('../submission/stack_'+str(cnt_re[1])+'.csv', index=None)