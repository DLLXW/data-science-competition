

#
from tqdm import tqdm
import os
import json
import numpy as np

out_file='./submit_roundB/submit_roundB_qyl.txt'
tuwen_results='./submit_roundB/submit_0422_merge_tuwen_roundB_v1.txt'#submit_0408_bert_tuwen submit_0409_macbert_tuwen
addr_results='./submit_roundB/submit_cnn_lstm_attr_roundB_v1.txt'#submit_cnn_lstm_attr submit_lstm_attr
submit=[]
tuwen_pre={}
#{"img_name":"test000255","match":{"图文":0,"领型":1,"袖长":1,"穿着方式":0}}
with open(tuwen_results, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        tuwen_pre[item['img_name']]=item['match']
#
tmp1=[]
tmp2=[]

with open(addr_results, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        tmp1.append(item['match']["图文"])
        item['match']["图文"]=tuwen_pre[item['img_name']]
        tmp2.append(item['match']["图文"])
        #print(tmp1,tmp2)
        submit.append(json.dumps(item, ensure_ascii=False)+'\n')
tmp1=np.array(tmp1)
tmp2=np.array(tmp2)
print(np.sum((tmp1!=tmp2)))
with open(out_file, 'w') as f:
    f.writelines(submit)