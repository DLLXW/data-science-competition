import os
from tqdm import tqdm
from helper import *
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def get_frt(df):
    #
    df=process_data(df)
    df = df.assign(DKFFE_bucket=pd.cut(df.DKFFE, [float('-inf'), 1e5, 1.5e5, 2e5, 2.5e5, 3e5, float('inf')],
                                       labels=[0, 1, 2, 3, 4, 5]))
    df['DKFFE_bucket'] = df['DKFFE_bucket'].astype(int)
    df = df.assign(DWJJLX_bucket=pd.cut(df.DWJJLX, [float('-inf'), 1e2, 2e2, 3e2, 4e2, 5e2, float('inf')],
                                        labels=[0, 1, 2, 3, 4, 5]))
    df['DWJJLX_bucket'] = df['DWJJLX_bucket'].astype(int)
    #
    df=deal_noise(df,'DKLL')
    df['ratio'] = df['GRYJCE'] / df['GRJCJS']
    df['ratio_bucket'] = (df['ratio'] * 100).astype(int)
    #
    cross_candidates = ['DWJJLX', 'DWSSHY', 'GRZHZT', 'XINGBIE', 'DKFFE_bucket', 'ratio_bucket']
    cross_lists = []
    cross_lists_tmp = []
    for can_i in cross_candidates:
        df[can_i + '_count'] = df[can_i].map(df[can_i].value_counts(dropna=True, normalize=True))
        for can_j in cross_candidates:
            cross_name = can_i + '_' + can_j
            if can_i != can_j and cross_name not in cross_lists_tmp:
                cross_lists.append(cross_name)
                cross_lists_tmp.append(can_i + '_' + can_j)
                cross_lists_tmp.append(can_j + '_' + can_i)
                df[cross_name] = df[can_i].astype(str) + '_' + df[can_j].astype(str)
                df[cross_name + '_count'] = df[cross_name].map(df[cross_name].value_counts(dropna=True, normalize=True))
    #
    cross_muti_candidates = ['ZHIYE', 'ZHICHEN', 'DWSSHY', 'DWJJLX', 'ratio_bucket']
    for can_i in cross_muti_candidates:
        df[can_i + '_count'] = df[can_i].map(df[can_i].value_counts(dropna=True, normalize=True))
        for can_j in cross_muti_candidates:
            for can_k in cross_muti_candidates:
                cross_name = can_i + '_' + can_j + '_' + can_k
                if can_i != can_j and can_i != can_k and can_j != can_k and cross_name not in cross_lists_tmp:
                    cross_lists.append(cross_name)
                    cross_lists_tmp.append(can_i + '_' + can_j + '_' + can_k)
                    cross_lists_tmp.append(can_i + '_' + can_k + '_' + can_j)
                    cross_lists_tmp.append(can_j + '_' + can_i + '_' + can_k)
                    cross_lists_tmp.append(can_j + '_' + can_k + '_' + can_i)
                    cross_lists_tmp.append(can_k + '_' + can_j + '_' + can_i)
                    cross_lists_tmp.append(can_k + '_' + can_i + '_' + can_j)
                    df[cross_name] = df[can_i].astype(str) + '_' + df[can_j].astype(str) + '_' + df[can_k].astype(str)
                    df[cross_name + '_count'] = df[cross_name].map(
                        df[cross_name].value_counts(dropna=True, normalize=True))
    #
    for frt in cross_lists:
        le = LabelEncoder()
        le.fit(list(df[frt].values))
        df[frt] = le.transform(list(df[frt].values))
    # print('build cross features finished................')
    # print(cross_lists)
    #
    df['ratio_div_dkll'] = df['ratio'] / df['DKLL']
    # df['re_ratio']=df['GRJCJS']/train_test['GRYJCE']
    df['income'] = df['GRJCJS'] * 12  # 缴存基数是指职工本人上一年的月平均工资
    # 当年提取额=个人账户上年结转+个人月缴存额*24-个人账户当年归集
    df['spend'] = df['GRZHSNJZYE'] + df['GRYJCE'] * 24 - df['GRZHDNGJYE']
    df['spend_div_income'] = df['spend'] / df['income']
    df['spend_sub_income'] = df['spend'] - df['income']
    #
    df['DKFFE_add_DKYE'] = df['DKFFE'] + df['DKYE']
    df['DKFFE_sub_DKYE'] = df['DKFFE'] - df['DKYE']  # 已还本金
    df['DKYE_div_DKFFE'] = df['DKYE'] / df['DKFFE']  # 未还本金/贷款发放额
    df['DKYE_div_DKYE_add_DKFFE'] = df['DKYE'] / df['DKFFE_add_DKYE']
    df['DKFFE_div_DKYE_add_DKFFE'] = df['DKFFE'] / df['DKFFE_add_DKYE']
    #
    num_frts=['GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRJCJS', 'DWYJCE', 'DKFFE', 'GRYJCE', 'DKYE', 'CSNY']
    num_frts += ['income', 'spend', 'spend_div_income', 'spend_sub_income', 'DKYE_div_DKFFE', 'DKYE_div_DKYE_add_DKFFE',
                 'DKFFE_div_DKYE_add_DKFFE']
    #
    df['DKYE_div_income'] = df['DKYE'] / df['income']  # 未还本金/收入
    df['DKFFE_div_income'] = df['DKFFE'] / df['income']
    df['DKFFE_DKYE_multi_DKLL'] = (df['DKFFE'] - df['DKYE']) * df['DKLL']
    df['DKFFE_multi_DKLL'] = df['DKFFE'] * df['DKLL']
    df['DKYE_multi_DKLL'] = df['DKYE'] * df['DKLL']
    df['GRZHDNGJYE_add_GRZHSNJZYE'] = df['GRZHDNGJYE'] + df['GRZHSNJZYE']
    df['GRZHDNGJYE_sub_GRZHSNJZYE'] = df['GRZHDNGJYE'] - df['GRZHSNJZYE']
    df['GRZHYE_sub_GRZHDNGJYE'] = df['GRZHYE'] - df['GRZHDNGJYE']  # 账户余额当年归集
    df['GRZHYE_div_GRZHDNGJYE'] = df['GRZHYE'] / df['GRZHDNGJYE']  # 账户余额/当年归集
    df['GRZHYE_sub_GRZHSNJZYE'] = df['GRZHYE'] - df['GRZHSNJZYE']
    df['GRZHYE_div_GRZHSNJZYE'] = df['GRZHYE'] / df['GRZHSNJZYE']
    df['GRZHYE_div_GRYJCE'] = df['GRZHYE'] / df['GRYJCE']
    df['DKYE_div_GRJCJS'] = df['DKYE'] / df['GRJCJS']
    df['DKYE_div_GRYJCE'] = df['DKYE'] / df['GRYJCE']
    #
    df['CSNY'].iloc[38354] = df['CSNY'].iloc[38354] // 1000
    df['CSNY'].iloc[52133] = df['CSNY'].iloc[52133] // 1000
    df['age'] = ((1612969200 - df['CSNY']) / (365 * 24 * 3600)).astype(int)
    num_frts += ['DKYE_div_income', 'DKFFE_DKYE_multi_DKLL', 'spend_div_income', 'spend_sub_income',
                 'GRZHDNGJYE_add_GRZHSNJZYE', 'age']
    candidate_col = num_frts.copy()
    # print('build base features finished................')
    ##sns.distplot(train_test['age'][train_test['age'] > 0])
    # df=df.assign(age_bucket=pd.cut(df.age, [float('-inf'),20,30,40,float('inf')],
    #                 labels=[0,1,2,3]))
    # df['age_bucket']=df['age_bucket'].astype(int)
    # #
    # df=df.assign(DKYE_bucket=pd.cut(df.DKYE, [float('-inf'),5e4,10e4,15e4,20e4,30e4,float('inf')],labels=[0,1,2,3,4,5]))
    # df['DKYE_bucket']=df['DKYE_bucket'].astype(int)
    # df=df.assign(DKFFE_bucket=pd.cut(df.DKFFE, [float('-inf'),5e4,10e4,20e4,30e4,float('inf')],labels=[0,1,2,3,4]))
    # df['DKFFE_bucket']=df['DKFFE_bucket'].astype(int)
    # df=df.assign(DKFFE_sub_DKYE_bucket=pd.cut(df.DKFFE_sub_DKYE, [float('-inf'),5e4,10e4,20e4,30e4,float('inf')],labels=[0,1,2,3,4]))
    # df['DKFFE_sub_DKYE_bucket']=df['DKFFE_sub_DKYE_bucket'].astype(int)
    # 类别特征的count特征
    #
    # cate_cols=['DWSSHY','GRZHZT']#
    # for f in tqdm(cate_cols):
    # df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique()))))
    # df[f + '_count'] = df[f].map(df[f].value_counts())
    # df = pd.concat([df,pd.get_dummies(df[f],prefix=f"{f}")],axis=1)
    #
    cnt_prob = ['ZHIYE_ZHICHEN_DWSSHY', 'ZHIYE_ZHICHEN_DWJJLX', 'GRZHZT', 'ratio_bucket']
    cate_cols_combine = [[cnt_prob[i], cnt_prob[j]] for i in range(len(cnt_prob)) for j in range(i + 1, len(cnt_prob))]

    for f1, f2 in tqdm(cate_cols_combine):
        df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['id'].transform('count')
        df['{}_in_{}_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / df[f2 + '_count']
        df['{}_in_{}_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / df[f1 + '_count']
    # print('build count_prop features finished................')
    #
    cate_frts = ['GRZHZT', 'ZHIYE_ZHICHEN_DWSSHY', 'ZHIYE_ZHICHEN_DWJJLX', 'ratio_bucket', 'is_month']
    num_frts = ['GRYJCE', 'GRJCJS', 'GRZHYE', 'DKFFE_div_DKYE_add_DKFFE', 'GRZHSNJZYE', 'GRZHDNGJYE']
    cate_frts += cnt_prob
    for f1 in tqdm(cate_frts):
        g = df.groupby(f1)
        for f2 in num_frts:
            for stat in ['sum', 'mean', 'max', 'min', 'std']:
                df['{}_{}_{}'.format(f1, f2, stat)] = g[f2].transform(stat)
    # num_frts = ['DKFFE','DKYE','DKFFE_sub_DKYE','DKLL','GRYJCE', 'GRJCJS','DKFFE_div_DKYE_add_DKFFE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'is_month']
    #
    num_frts = ['GRYJCE', 'GRZHDNGJYE', 'GRZHYE_sub_GRZHSNJZYE', 'DKFFE_sub_DKYE', 'DKFFE_div_DKYE_add_DKFFE',
                'GRZHSNJZYE']
    cate_frts = ['GRZHZT', 'ZHIYE_ZHICHEN_DWSSHY', 'ZHIYE_ZHICHEN_DWJJLX', 'DWJJLX', 'DKLL', 'ratio_bucket']
    for f1 in tqdm(cate_frts):
        for f2 in num_frts:
            tmp = df.groupby(f1)[f2].agg([sum, min, max, np.mean]).reset_index()
            tmp = pd.merge(df, tmp, on=f1, how='left')
            df['{}_sub_sum_{}'.format(f2, f1)] = df[f2] / tmp['sum']
            df['{}_sub_mean_{}'.format(f2, f1)] = df[f2] - tmp['mean']
            df['{}_sub_min_{}'.format(f2, f1)] = df[f2] - tmp['min']
            df['{}_sub_max_{}'.format(f2, f1)] = df[f2] - tmp['max']
    #
    num_frts = ['DKFFE_sub_DKYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'GRJCJS', 'DKFFE', 'DKYE']
    for f1 in tqdm(num_frts):
        g = df.groupby(f1)
        for f2 in num_frts:
            if f1 != f2:
                for stat in ['sum', 'std', 'min']:
                    df['{}_{}_{}'.format(f1, f2, stat)] = g[f2].transform(stat)

    for i in range(len(candidate_col)):
        for j in range(i, len(candidate_col)):
            col_i = candidate_col[i]
            col_j = candidate_col[j]
            df['{}_add_{}'.format(col_i, col_j)] = df[col_i] + df[col_j]
            df['{}_sub_{}'.format(col_i, col_j)] = df[col_i] - df[col_j]
            df['{}_mul_{}'.format(col_i, col_j)] = df[col_i] * df[col_j]
            df['{}_div_{}'.format(col_i, col_j)] = df[col_i] / df[col_j]
    return df
#
if __name__=="__main__":
    #
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    train_test = pd.concat((train_df, test_df)).reset_index(drop=True)
    print(train_df.shape, test_df.shape, train_test.shape)
    train_test = get_frt(train_test)[['id']+select_frts+['label']]
    print(train_test)
    if not os.path.exists('tmps/'):
        os.makedirs('tmps/')
    train_test.to_csv('tmps/total_frts.csv',index=False)
