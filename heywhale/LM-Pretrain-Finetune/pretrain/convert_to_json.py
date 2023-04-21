import pandas as pd
import os
import json
from tqdm import tqdm
df=pd.read_csv('/home/trojanjet/project/weiqin/diag/bart/data/diagnosis/train.csv',header=None)
df.columns=['id','des','diag']
data=df['des'].values.tolist()+df['diag'].values.tolist()
with open('./diag_train_v1.json','w') as w:
    for line in tqdm(df['des'].values):
        if len(line)>15:
            line=json.dumps({"text":line},ensure_ascii=False)
            w.write(line+'\n')
w.close()
# w=open('diag_vocab.txt','a')
# for i in range(1,1301):
#     w.write(str(i+100)+'\n')
# w.close()