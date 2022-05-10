import pandas as pd
import lightgbm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from helper import tpr_weight_funtion
from sklearn.metrics import roc_curve, auc, roc_auc_score,log_loss
from helper import select_frts,get_latest_sub
import os
def tpr_eval_score(y_pre, data):
    y_true = data.get_label()
    tpr=tpr_weight_funtion(y_true, y_pre)
    return 'tpr', tpr, True,
def lgb_cv(train_x, train_y, test_x, class_num=1):
    predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    folds = 10
    seed = 2021
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
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
            'objective': 'binary',
            #'scale_pos_weight':0.05,
            'metrics':'auc',#'binary_logloss',
            'num_leaves': 2 ** 6-1,
            #'lambda_l2': 10,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'learning_rate': 0.05,
            'seed': 2020,
            'nthread': 8,
            'num_class': class_num,
            'verbose': -1,
        }
        num_round = 4000
        early_stopping_rounds = 200
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
            pre = model.predict(te_x, num_iteration=model.best_iteration)#(8000,)
            pred = model.predict(test_x, num_iteration=model.best_iteration)#(15000,)
            train[test_index] = pre
            test_pre[i, :] = pred
            cv_scores.append(roc_auc_score(te_y, pre))
            eval_score=tpr_weight_funtion(te_y,pre)
            tpr_scores.append(eval_score)
            cv_rounds.append(model.best_iteration)
            test_pre_all[i, :] = pred
        #
        print("cv_score is:", cv_scores)
        print("tpr-score is:", tpr_scores)
    use_mean=True
    if use_mean:
        test[:] = test_pre.mean(axis=0)
    else:
        pass
    #
    print("val_mean:" , np.mean(cv_scores), np.mean(tpr_scores))
    print("val_std:", np.std(tpr_scores))
    return train, test, test_pre_all, np.mean(cv_scores),np.mean(tpr_scores),importance_list


if __name__=="__main__":
    # -------------------
    importance_select = select_frts
    train_test=pd.read_csv('tmps/total_frts.csv')
    drop_feats = [f for f in train_test.columns if train_test[f].nunique() == 1 or train_test[f].nunique() == 0]
    # len(drop_feats), drop_feats
    train_df = train_test[:40000]
    test_df = train_test[40000:].reset_index(drop=True)
    # train_df.shape,test_df.shape
    cate_cols = ['XINGBIE', 'HYZK', 'ZHIYE', 'ZHICHEN', 'ZHIWU', 'XUELI', 'DWJJLX', 'DWSSHY']
    select_frts = [f for f in train_df.columns if f not in drop_feats + ['id', 'label', 'CSNY']]
    print('len(select_frts):', len(select_frts))
    # select_frts=important_frt[:50]
    cate_cols = list(set(select_frts).intersection(set(cate_cols)))
    # ---------------
    use_lgb = True
    if use_lgb:
        select_frts = importance_select[:100]  # 去掉target_encoding
        selct_frts_sample = []
        for i in range(len(select_frts)):
            if i < 10:
                selct_frts_sample.append(select_frts[i])
            elif i % 2 != 0:
                selct_frts_sample.append(select_frts[i])
        #
        train_x = train_df[select_frts].copy()
        train_y = train_df["label"]
        print(train_x.shape)
        test_x = test_df[select_frts].copy()
        lgb_train, lgb_test, sb, cv_scores, tpr_scores, importance_list = lgb_cv(train_x, train_y, test_x)
        # print(importance_list[:100])
        submit = pd.read_csv('data/submit.csv')
        submit['label'] = lgb_test
        if not os.path.exists('submit/'):
            os.makedirs('submit/')
        save_dir = 'submit/' + 'tpr_' + str(int(1000 * tpr_scores)) + 'auc_' + str(int(1000 * cv_scores)) + '.csv'
        submit=get_latest_sub(submit)
        submit.to_csv(save_dir, index=False)
        print(save_dir)
    else:
        print('using catboost...............')
        #tpr, val_aucs, catboost_pre = cab(train_df, test_df, select_frts, cate_cols=None)
        #print(tpr, val_aucs)