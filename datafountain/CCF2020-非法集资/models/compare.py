import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from utils import cnt_result
sub1=pd.read_csv("D:/非法集资/pycharm_project/submission/lgb_pseudo_847_934.csv")['score']
sub2=pd.read_csv(r"D:\非法集资\pycharm_project\submission\submit_stack_cab_lgb_xgb897.csv")['score']
sub3=np.sqrt(sub1*sub2)
#sub3=0.5*sub1+0.5*sub2
sub1 = (sub1 > 0.5).astype(int)
sub2 = (sub2 > 0.5).astype(int)
sub3 = (sub3 > 0.5).astype(int)


cnt_re = cnt_result(sub1)
print("合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
cnt_re = cnt_result(sub2)
print("合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
cnt_re = cnt_result(sub3)
print("合法的数量%d;违法的数量%d,合法/违法%f" % (cnt_re[0], cnt_re[1], cnt_re[0] / cnt_re[1]))
print(accuracy_score(sub1,sub2))
print(accuracy_score(sub1,sub3))
print(accuracy_score(sub2,sub3))