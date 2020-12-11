import catboost
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from utils import eval_score,select_frt_names,cnt_result
from sklearn.model_selection import StratifiedKFold
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
def k_fold_serachParmaters(model, train_val_data,
                           train_val_kind, test_kind=None, mode='Train'):
    mean_f1 = {'F': 0, 'valid_f1': 0, 'valid_p': 0, 'valid_r': 0}
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    if mode != 'Train':
        pred_Test = np.zeros(len(test_kind))
    else:
        pred_Test = None
    for train, test in sk.split(train_val_data, train_val_kind):
        x_train = train_val_data.iloc[train]
        y_train = train_val_kind.iloc[train]
        x_test = train_val_data.iloc[test]
        y_test = train_val_kind.iloc[test]

        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        val_score = eval_score(y_test, pred)

        mean_f1['F'] += val_score['F'] / n_splits
        mean_f1['valid_f1'] += val_score['valid_f1'] / n_splits
        mean_f1['valid_p'] += val_score['valid_p'] / n_splits
        mean_f1['valid_r'] += val_score['valid_r'] / n_splits
        if mode != 'Train':
            pred_Test += model.predict_proba(test_kind)[:, 1] / n_splits
    return mean_f1, pred_Test

def grid_search():
    model = catboost.CatBoostClassifier(silent=True)
    param_dist = {"depth": [6, 8],
                  "learning_rate": [0.04, 0.06],
                  "iterations": [70, 100,]
                  }

    grid_search = GridSearchCV(model, n_jobs=-1, param_grid=param_dist, cv=5, scoring='f1', verbose=5)
    grid_search.fit(train_data[select_features],
                    train_data['label'])
    return grid_search.best_estimator_, grid_search.best_score_
def cab_tta(tta_fold = 20):
    score_tta = None
    score_list_F = []
    score_list_f1 = []
    score_list_p = []
    score_list_r = []
    lr_choice=[0.05,0.07,0.09,0.1]
    for _ in tqdm(range(tta_fold)):
        params = {  'iterations':np.random.randint(50, 80),
                    'learning_rate':random.choice(lr_choice),
                    'depth': np.random.randint(8, 11),
                    'silent': True,
                    'thread_count': 8,
                    'task_type': 'CPU',
                    'cat_features':cat_features,
                    #'early_stopping_rounds': 100,
                      }
        train_data[cat_features]=train_data[cat_features].fillna(-1).astype(int)
        test_data[cat_features] = test_data[cat_features].fillna(-1).astype(int)
        clf = catboost.CatBoostClassifier(**params)

        score, test_pred = k_fold_serachParmaters(
                                                  clf,
                                                  train_data[select_features],
                                                  train_data['label'],
                                                  test_data[select_features],
                                                  mode='test'
                                                  )

        if score_tta is None:
            score_tta = test_pred / tta_fold
        else:
            score_tta += test_pred / tta_fold
        # print(score)
        score_list_F.append(score['F'])
        score_list_f1.append(score['valid_f1'])
        score_list_p.append(score['valid_p'])
        score_list_r.append(score['valid_r'])

    print("F:", np.array(score_list_F).mean(), np.array(score_list_F).std())
    print("f1:", np.array(score_list_f1).mean(), np.array(score_list_f1).std())
    print("p:", np.array(score_list_p).mean(), np.array(score_list_p).std())
    print("r", np.array(score_list_r).mean(), np.array(score_list_r).std())
    #
    return score_tta,np.array(score_list_f1).mean()

if __name__=="__main__":
    #获取特征
    train_data=pd.read_csv("../usr_data/train_data.csv")
    test_data=pd.read_csv("../usr_data/test_data.csv")
    #特征筛选
    select_features ,cat_features=select_frt_names(train_data,thres=0.1)
    #
    score_tta,f_1=cab_tta(10)

    cnt_re = cnt_result(score_tta)
    print("合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    test_data_sub = test_data.copy()
    score_tta = (score_tta > 0.5).astype(int)
    test_data_sub['score'] = score_tta
    test_data_sub[['id', 'score']].to_csv('../submission/cab_'+ str(int(1000*f_1))+'_'+str(cnt_re[1])+'.csv', index=None)
    # best_estimator_, best_score_=grid_search()
    # print(best_estimator_)
    # print(best_score_)
