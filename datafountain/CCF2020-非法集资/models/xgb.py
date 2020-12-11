import xgboost
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from utils import eval_score,select_frt_names,cnt_result
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def k_fold_serachParmaters(model, train_val_data,
                           train_val_kind, test_frt=None, mode='Train',cat_features=None):
    mean_f1 = {'F': 0, 'valid_f1': 0, 'valid_p': 0, 'valid_r': 0}
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    if mode != 'Train':
        pred_Test = np.zeros(len(test_frt))
    else:
        pred_Test = None
    for train, test in sk.split(train_val_data, train_val_kind):
        x_train = train_val_data.iloc[train]
        y_train = train_val_kind.iloc[train]
        x_test = train_val_data.iloc[test]
        y_test = train_val_kind.iloc[test]

        model.fit(x_train, y_train,
                      eval_set=[(x_test, y_test)],
                      early_stopping_rounds=100,
                      verbose=False)

        pred = model.predict(x_test)
        val_score = eval_score(y_test, pred)

        mean_f1['F'] += val_score['F'] / n_splits
        mean_f1['valid_f1'] += val_score['valid_f1'] / n_splits
        mean_f1['valid_p'] += val_score['valid_p'] / n_splits
        mean_f1['valid_r'] += val_score['valid_r'] / n_splits
        if mode != 'Train':
            pred_Test += model.predict_proba(test_frt)[:, 1] / n_splits
    return mean_f1, pred_Test
def grid_search():
    lg = xgboost.XGBClassifier(silent=False)
    param_dist = {"max_depth": [6, 8, 10],
                  "learning_rate": [0.04, 0.06, 0.07],
                  "num_leaves": [5, 7, 8, 10],
                  "min_child_samples": [5, 10, 15],
                  "n_estimators": [70, 120, 150]
                  }

    grid_search = GridSearchCV(lg, n_jobs=8, param_grid=param_dist, cv=5, scoring='f1', verbose=5)
    grid_search.fit(train_data.drop(['id', 'label'], axis=1)[select_features],
                    train_data['label'], categorical_feature=cat_features, )
    return grid_search.best_estimator_, grid_search.best_score_
def xgb_tta(tta_fold = 20):
    score_tta = None
    score_list_F = []
    score_list_f1 = []
    score_list_p = []
    score_list_r = []
    #
    for _ in tqdm(range(tta_fold)):
        clf = xgboost.XGBClassifier(
            max_depth=np.random.randint(6, 10),
            min_child_weight=np.random.randint(1, 5),
            learning_rate=0.05,
            n_jobs=8,
                )

        score, test_pred = k_fold_serachParmaters(
                                                  clf,
                                                  train_data[select_features],
                                                  train_data['label'],
                                                  test_data[select_features],
                                                  mode='test',
                                                  cat_features=cat_features
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
    # #特征筛选
    select_features ,cat_features=select_frt_names(train_data,thres=0.1)
    #
    score_tta,f_1=xgb_tta(20)
    cnt_re = cnt_result(score_tta)
    print("合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    test_data_sub = test_data.copy()
    score_tta = (score_tta > 0.5).astype(int)
    test_data_sub['score'] = score_tta
    test_data_sub[['id', 'score']].to_csv('../submission/xgb_' + str(int(1000*f_1))+'_'+ str(cnt_re[1]) + '.csv', index=None)

