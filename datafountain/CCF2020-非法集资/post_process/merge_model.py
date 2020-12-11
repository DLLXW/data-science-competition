import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
#from utils import cnt_result
def cnt_result(x):
    cnt_re={0:0,1:0}
    for a in x:
        if a<=0.5:
            cnt_re[0]+=1
        else:
            cnt_re[1]+=1
    return cnt_re
def get_jiaoji(re1,re2):
    re3_geom=np.sqrt(re1*re2)#几何平均
    re3_mean=0.5*re1+0.5*re2#算术平均
    #交集
    jiaoji_list=[]
    for i in range(len(re1)):
        if re1[i]>0.5 and re2[i]>0.5:
            jiaoji_list.append(max(re1[i],re2[i]))
        else:
            jiaoji_list.append(min(re1[i], re2[i]))
    jiaoji_list=np.array(jiaoji_list)
    re_last = 0.5 * jiaoji_list + 0.3 * re3_geom + 0.2 * re3_mean
    cnt_re = cnt_result(jiaoji_list)
    print("交集合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    cnt_re = cnt_result(re3_mean)
    print("算术平均合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    cnt_re = cnt_result(re3_geom)
    print("几何平均合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    cnt_re = cnt_result(re_last)
    print("最终融合合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
    re3_mean_int= (re3_mean > 0.5).astype(int)
    re3_geom_int = (re3_geom > 0.5).astype(int)
    re_last_int = (re_last > 0.5).astype(int)
    return re_last,cnt_re[1]


#sub_df1=pd.read_csv("D:/非法集资/pycharm_project/submission/lgb_pseudo_847_934.csv")#842
#sub_df2=pd.read_csv(r"D:\非法集资\pycharm_project\submission\submit_stack_cab_lgb_xgb897.csv")#842
sub_df1=pd.read_csv("../submission/847_dy.csv")#842
sub_df2=pd.read_csv("../submission/lgb_121_905_889.csv")#842
sub1=sub_df1['score'].values
sub2=sub_df2['score'].values
re_last,cnt=get_jiaoji(sub1,sub2)
sub_merge=sub_df1.copy()
sub_merge['score']=re_last
sub_merge.to_csv('../submission/submit_merge'+str(cnt)+'_1204.csv',index=False)