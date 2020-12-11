from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)[:, 1]

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


# "-----------------定义oof:stacking的核心流程-------------"
print("-------->>>>>定义oof:stacking的核心流程<<<<<--------")
'''
'''


def get_oof(clf, x_train, y_train, x_test,ntrain,ntest,NFOLDS=5):
    oof_train = np.zeros((ntrain,))  # (14865,)
    oof_test = np.zeros((ntest,))  # (10000,)
    oof_test_skf = np.empty((NFOLDS, ntest))  # (5, 10000)
    #
    fold = 0
    sk = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2020)
    for train_index, test_index in sk.split(x_train, y_train):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]  # 该折余下的验证集

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)  # 对该折余下的验证集做预测，将结果填补在这些数据在原始数据中的位置
        oof_test_skf[fold, :] = clf.predict_proba(x_test)  # 用此时的模型（第i折的模型）对测试做预测，放在第i折对应的位置
        # oof_train[test_index] = clf.predict(x_te)
        # oof_test_skf[fold, :] = clf.predict(x_test)
        fold += 1

    oof_test[:] = oof_test_skf.mean(axis=0)  # 将这N折模型对测试集的预测结果进行一个平均，作为改模型的预测结果
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def eval_score(y_test, y_pre):
    valid_f1 = f1_score(y_test, y_pre)
    valid_p = precision_score(y_test, y_pre)
    valid_r = recall_score(y_test, y_pre)
    F = valid_p * 0.7 + valid_r * 0.2 + valid_f1 * 0.1
    return {'F': F, 'valid_f1': valid_f1, 'valid_p': valid_p, 'valid_r': valid_r}

def select_frt_names(train_data, thres=0.1):
    tmp = train_data.corr()['label'].abs()[train_data.corr()['label'].abs() > thres].to_dict()
    sort_corr = sorted(tmp.items(), key=lambda x: x[1], reverse=True)
    select_features = [per[0] for per in sort_corr[1:]]
    # 按照特征和label相关性进行筛选
    cat_features = ['oplocdistrict', 'industryphy', 'industryco', 'enttype',
                    'enttypeitem', 'enttypeminu', 'enttypegb',
                    'oploc', 'opform',
                    'dom_prefix', 'id_prefix',
                    'industryphy_industryco',
                    'enttype_enttypeitem', 'enttypegb_enttypeminu',
                    'industryphy_enttype', 'enttype_enttypeitem_industryphy_industryco'
                    ]
    cat_features = list(set(select_features).intersection(set(cat_features)))
    return select_features, cat_features
#
def cnt_result(x):
    cnt_re={0:0,1:0}
    for a in x:
        if a<=0.5:
            cnt_re[0]+=1
        else:
            cnt_re[1]+=1
    return cnt_re