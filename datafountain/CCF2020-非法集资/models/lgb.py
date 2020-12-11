#
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import numpy as np
import pandas as pd
from utils import eval_score,select_frt_names,cnt_result
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

def intersection(re1,re2):
    # 交集
    inters_list = []
    for i in range(len(re1)):
        if re1[i] > 0.5 and re2[i] > 0.5:
            inters_list.append(max(re1[i], re2[i]))
        else:
            inters_list.append(min(re1[i], re2[i]))
    return np.array(inters_list)
def k_fold_serachParmaters(model, train_val_data,
                           train_val_kind, test_frt=None, mode='Train',cat_features=None):
    mean_f1 = {'F': 0, 'valid_f1': 0, 'valid_p': 0, 'valid_r': 0}
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    if mode != 'Train':
        pred_Test = np.zeros(len(test_frt))
    else:
        pred_Test = None
    cnt=0
    for train, test in sk.split(train_val_data, train_val_kind):
        cnt+=1
        x_train = train_val_data.iloc[train]
        y_train = train_val_kind.iloc[train]
        x_test = train_val_data.iloc[test]
        y_test = train_val_kind.iloc[test]

        model.fit(x_train, y_train,
                      eval_set=[(x_test, y_test)],
                      categorical_feature=cat_features,
                      early_stopping_rounds=100,
                      verbose=False)

        pred = model.predict(x_test)
        val_score = eval_score(y_test, pred)

        mean_f1['F'] += val_score['F'] / n_splits
        mean_f1['valid_f1'] += val_score['valid_f1'] / n_splits
        mean_f1['valid_p'] += val_score['valid_p'] / n_splits
        mean_f1['valid_r'] += val_score['valid_r'] / n_splits
        if mode != 'Train':
            if use_intersection_cv:
                if cnt==1:
                    pred_Test=model.predict_proba(test_frt)[:, 1]
                else:
                    pred_Test=intersection(pred_Test,model.predict_proba(test_frt)[:, 1])
            else:
                pred_Test += model.predict_proba(test_frt)[:, 1] / n_splits
    return mean_f1, pred_Test
def grid_search():
    lg = lgb.LGBMClassifier(silent=False)
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
def lgb_tta(tta_fold = 20):
    score_tta = None
    score_list_F = []
    score_list_f1 = []
    score_list_p = []
    score_list_r = []
    lr_choice = [0.05, 0.07, 0.09, 0.1]
    for _ in tqdm(range(tta_fold)):
        clf = lgb.LGBMClassifier(
            num_leaves=np.random.randint(6, 10), min_child_samples=np.random.randint(2, 5),
            max_depth=np.random.randint(5, 10), learning_rate=random.choice(lr_choice),
            n_estimators=150, n_jobs=-1)

        score, test_pred = k_fold_serachParmaters(
                                                  clf,
                                                  train_data[select_features],
                                                  train_data['label'],
                                                  test_data[select_features],
                                                  mode='test',
                                                  cat_features=cat_features
                                                  )
        if use_intersection:
            if score_tta is None:
                score_tta = test_pred
            else:
                score_tta = intersection(score_tta, test_pred)
        else:
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
    select_features ,cat_features=select_frt_names(train_data,thres=0.2)
    print(len(select_features))
    #
    use_intersection=False#是否在随机种子融合时使用交集，2个千，tta30次
    use_intersection_cv=True#是否在五折cv的地方使用交集，3个千，tta30次
    score_tta,f_1=lgb_tta(10)
    cnt_re = cnt_result(score_tta)
    print("合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    test_data_sub = test_data.copy()
    score_tta_int = (score_tta > 0.5).astype(int)
    test_data_sub['score'] = score_tta
    #test_data_sub[['id', 'score']].to_csv('../submission/lgb_pseudo_' + str(int(1000*f_1))+'_'+ str(cnt_re[1]) + '.csv', index=None)
    test_data_sub[['id', 'score']].to_csv('../submission/lgb_intersect_' + str(int(1000 * f_1)) + '_' + str(cnt_re[1]) + '.csv', index=None)

