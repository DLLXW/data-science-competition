

import pandas as pd
import numpy as np

def process_data(df):
    num_frts = ['GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRJCJS', 'DWYJCE', 'DKFFE', 'GRYJCE', 'DKYE', 'CSNY']
    for frt in num_frts:
        df[frt] -= 237
    return df
def tpr_weight_funtion(y_true,y_predict):
    d= pd.DataFrame()
    d['prob']=list(y_predict)
    d['y']=list(y_true)
    d=d.sort_values(['prob'], ascending=[0])
    y=d.y
    PosAll=pd.Series(y).value_counts()[1]
    NegAll=pd.Series(y).value_counts()[0]
    pCumsum=d['y'].cumsum()
    nCumsum=np.arange(len(y))-pCumsum+1
    pCumsumPer=pCumsum/PosAll
    nCumsumPer=nCumsum/NegAll
    TR1=pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2=pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3=pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3
def deal_noise(df,col):
    noise_ratio = sorted([round(2.75*10/12,3),round(2.75*1.1*10/12,3),2.75,round(3.25*10/12,3),round(2.75*1.1,3),round(3.25*1.1*10/12,3),3.25,3.25*1.1])#噪声利率的分布范围
    assign_slice = []
    for i in range(len(noise_ratio) - 1):
        cur = noise_ratio[i]
        next_point = noise_ratio[i + 1]
        assign_slice.append((cur + next_point) / 2)

    def get_r(r, assign_slice):
        for i in range(len(assign_slice)):
            if assign_slice[i] > r:
                return noise_ratio[i]
        return noise_ratio[-1]

    restore_DKLL = []
    for r in df[col].values:
        restore_DKLL.append(get_r(r, assign_slice))
    #
    for i in range(len(restore_DKLL)):
        if restore_DKLL[i] == 2.292:
            restore_DKLL[i] = 2.75
        elif restore_DKLL[i] == 2.521:
            restore_DKLL[i] = 2.75 * 1.1
        elif restore_DKLL[i] == 2.708:
            restore_DKLL[i] = 3.25
        elif restore_DKLL[i] == 2.979:
            restore_DKLL[i] = 3.25 * 1.1
    df[col] = restore_DKLL
    df['is_month'] = 0
    for p in [2.292,2.708,2.521,2.979]:
        df['is_month'].loc[df[col]==p]=1
    return df

def get_latest_sub(df):
    df_2=df.copy()
    df_1=pd.read_csv('tmps/roundB.csv')
    rank_info = (0.95 * df_1['label'].rank(ascending=1, method='first') + 0.05 * df_2['label'].rank(ascending=1,method='first'))
    df['label'] = rank_info/len(df)
    return df

select_frts=['ZHIYE_ZHICHEN_DWJJLX_DKFFE_div_DKYE_add_DKFFE_mean',
 'DWSSHY',
 'ZHIYE_ZHICHEN_DWJJLX_GRZHDNGJYE_sum',
 'DKFFE_div_DKYE_add_DKFFE_sub_sum_GRZHZT',
 'DWYJCE_div_age',
 'ratio_bucket_in_ZHIYE_ZHICHEN_DWJJLX_prop',
 'GRZHZT_GRZHYE_mean',
 'GRJCJS_div_DWYJCE',
 'ZHIYE_ZHICHEN_DWSSHY_DKFFE_div_DKYE_add_DKFFE_mean',
 'ZHIYE_ZHICHEN_DWJJLX_GRZHYE_std',
 'GRZHZT_in_ZHIYE_ZHICHEN_DWJJLX_prop',
 'ratio',
 'GRYJCE_sub_sum_ZHIYE_ZHICHEN_DWJJLX',
 'GRZHZT',
 'ZHIYE_ZHICHEN_DWSSHY_in_ZHIYE_ZHICHEN_DWJJLX_prop',
 'ZHIYE_ZHICHEN_DWJJLX_GRZHDNGJYE_mean',
 'GRZHZT_in_ZHIYE_ZHICHEN_DWSSHY_prop',
 'GRYJCE_sub_mean_DWJJLX',
 'income_div_age',
 'DWJJLX_DWSSHY',
 'DWJJLX_DWSSHY_count',
 'DWSSHY_DWJJLX_ratio_bucket',
 'DKFFE_div_DKYE_add_DKFFE_sub_max_GRZHZT',
 'DKFFE_GRZHDNGJYE_sum',
 'DWSSHY_GRZHZT_count',
 'ZHIYE_DWSSHY_DWJJLX',
 'DKFFE_div_DKYE_add_DKFFE_sub_min_ZHIYE_ZHICHEN_DWJJLX',
 'DWSSHY_DWJJLX_ratio_bucket_count',
 'GRYJCE_sub_sum_DWJJLX',
 'ZHIYE_ZHICHEN_DWJJLX_GRZHDNGJYE_max',
 'DKFFE_sub_DKYE_sub_max_GRZHZT',
 'DKFFE_sub_DKYE_sub_sum_GRZHZT',
 'GRYJCE_sub_mean_ZHIYE_ZHICHEN_DWSSHY',
 'ratio_div_dkll',
 'DKYE_div_DKFFE',
 'DKFFE_DKYE_multi_DKLL_div_age',
 'ZHIYE_ZHICHEN_DWSSHY_GRZHZT_count',
 'CSNY_mul_DKFFE_DKYE_multi_DKLL',
 'DWSSHY_ratio_bucket_count',
 'GRYJCE_sub_max_ZHIYE_ZHICHEN_DWSSHY',
 'GRYJCE_sub_sum_ZHIYE_ZHICHEN_DWSSHY',
 'GRZHYE_sub_GRZHSNJZYE_sub_min_DWJJLX',
 'ZHIYE_ZHICHEN_DWJJLX_in_ZHIYE_ZHICHEN_DWSSHY_prop',
 'GRJCJS_div_age',
 'GRJCJS_GRYJCE_min',
 'ZHIYE_DWSSHY_ratio_bucket_count',
 'ZHIYE_ZHICHEN_DWSSHY_in_ratio_bucket_prop',
 'GRYJCE_sub_mean_DKLL',
 'DWSSHY_ratio_bucket',
 'DKYE_div_DKFFE_DKYE_multi_DKLL',
 'DKYE_div_DKFFE_div_DKYE_div_DKYE_add_DKFFE',
 'GRZHYE_sub_GRZHSNJZYE_sub_max_GRZHZT',
 'ZHIYE_ZHICHEN_DWSSHY_GRYJCE_mean',
 'DKYE_div_DKFFE_sub_DKFFE_div_DKYE_add_DKFFE',
 'ZHIYE_DWSSHY_DWJJLX_count',
 'DKFFE_div_DKYE_add_DKFFE_sub_mean_DKLL',
 'GRYJCE_sub_max_ratio_bucket',
 'GRJCJS_mul_CSNY',
 'GRZHZT_DKFFE_bucket',
 'ZHICHEN_DWSSHY_DWJJLX',
 'GRJCJS_div_DKFFE_DKYE_multi_DKLL',
 'GRZHDNGJYE_div_GRJCJS',
 'ZHIYE_ZHICHEN_DWSSHY_GRZHSNJZYE_max',
 'GRZHDNGJYE_sub_min_ZHIYE_ZHICHEN_DWJJLX',
 'CSNY_add_DKFFE_DKYE_multi_DKLL',
 'ZHIYE_ZHICHEN_DWSSHY_GRZHDNGJYE_mean',
 'GRZHZT_DKFFE_div_DKYE_add_DKFFE_max',
 'ZHIYE_ZHICHEN_DWJJLX_GRZHYE_mean',
 'GRYJCE_sub_sum_ratio_bucket',
 'ZHIYE_ZHICHEN_DWSSHY_GRJCJS_mean',
 'ZHIYE_ZHICHEN_DWSSHY_GRYJCE_sum',
 'GRZHDNGJYE_div_DWYJCE',
 'GRYJCE_div_age',
 'GRZHSNJZYE_sub_max_GRZHZT',
 'ZHIYE_ZHICHEN_DWSSHY_GRZHDNGJYE_sum',
 'GRZHDNGJYE_sub_GRJCJS',
 'DKFFE_div_DKYE_add_DKFFE_sub_max_ZHIYE_ZHICHEN_DWSSHY',
 'ZHIYE_DWJJLX_ratio_bucket',
 'ZHIYE_DWSSHY_ratio_bucket',
 'DWJJLX_GRZHZT_count',
 'DKFFE_sub_DKYE_DKFFE_sum',
 'ZHIYE_ZHICHEN_DWJJLX_in_ratio_bucket_prop',
 'ZHIYE_ZHICHEN_DWJJLX_GRZHZT_count',
 'DKYE_div_DKFFE_add_DKYE_div_DKFFE',
 'ZHIYE_ZHICHEN_DWSSHY_GRZHDNGJYE_max',
 'DWYJCE_sub_income',
 'DWYJCE_mul_CSNY',
 'ZHIYE_ZHICHEN_DWSSHY_count',
 'GRZHDNGJYE_sub_max_GRZHZT',
 'ratio_bucket_in_ZHIYE_ZHICHEN_DWSSHY_prop',
 'GRZHDNGJYE_GRYJCE_min',
 'GRYJCE_sub_sum_GRZHZT',
 'DWSSHY_DKFFE_bucket',
 'GRJCJS_div_GRYJCE',
 'DWYJCE_mul_DKFFE_DKYE_multi_DKLL',
 'GRYJCE_GRJCJS_std',
 'GRYJCE_sub_min_ZHIYE_ZHICHEN_DWSSHY',
 'spend_mul_DKFFE_DKYE_multi_DKLL',
 'ZHIYE_ZHICHEN_DWJJLX_DKFFE_div_DKYE_add_DKFFE_std',
 'GRYJCE_GRJCJS_sum',
 'GRZHYE_sub_DWYJCE',
 'ZHIYE_ZHICHEN_DWSSHY_DKFFE_div_DKYE_add_DKFFE_std',
 'ZHIYE_ZHICHEN_DWJJLX_GRYJCE_std',
 'DWJJLX_ratio_bucket',
 'GRZHYE_sub_income',
 'DWYJCE_div_income',
 'GRYJCE_sub_mean_GRZHZT',
 'GRZHDNGJYE_GRJCJS_min',
 'GRYJCE_sub_mean_ZHIYE_ZHICHEN_DWJJLX',
 'DKYE_div_DKFFE_div_DKFFE_DKYE_multi_DKLL',
 'GRJCJS_mul_DKYE_div_DKFFE',
 'GRYJCE_DKFFE_min',
 'DKFFE_div_DKYE_add_DKFFE_sub_sum_DKLL',
 'GRJCJS_mul_DKFFE',
 'ZHICHEN_DWSSHY_ratio_bucket',
 'GRYJCE_div_DKYE_div_DKFFE',
 'ZHIYE_DWJJLX_ratio_bucket_count',
 'DWJJLX_XINGBIE',
 'GRZHSNJZYE_sub_DWYJCE',
 'DWSSHY_GRZHZT',
 'GRZHDNGJYE_add_CSNY',
 'DWSSHY_count',
 'DWJJLX_DKFFE_bucket',
 'ZHIYE_ZHICHEN_DWSSHY_DKFFE_div_DKYE_add_DKFFE_sum',
 'DKFFE_GRJCJS_min',
 'GRYJCE_DKFFE_sum',
 'GRZHDNGJYE_add_income',
 'ZHICHEN_DWSSHY_ratio_bucket_count',
 'GRZHSNJZYE_mul_age',
 'GRYJCE_mul_DKFFE_DKYE_multi_DKLL',
 'ZHIYE_ZHICHEN_DWJJLX_GRYJCE_mean',
 'GRJCJS_mul_age',
 'GRZHZT_GRZHDNGJYE_max',
 'DKFFE_sub_CSNY',
 'ZHIYE_ZHICHEN_DWSSHY_GRYJCE_std',
 'income_add_spend',
 'DKFFE_sub_spend_div_income',
 'ZHICHEN_DWSSHY_DWJJLX_count',
 'DWYJCE_add_DKFFE',
 'GRZHSNJZYE_DKFFE_std',
 'GRZHZT_count',
 'GRZHZT_DKFFE_bucket_count',
 'GRJCJS_sub_income',
 'GRZHZT_in_ratio_bucket_prop',
 'ratio_bucket_GRJCJS_std',
 'DKYE_div_DKYE_div_income',
 'CSNY_sub_GRZHDNGJYE_add_GRZHSNJZYE',
 'DWYJCE_sub_DKYE_div_income',
 'DKFFE_div_DKYE_add_DKFFE_mul_DKFFE_DKYE_multi_DKLL',
 'GRZHYE_sub_GRYJCE']