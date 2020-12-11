import pandas as pd

df = pd.read_csv('../submission/submit_8485.csv')
pres=df['score'].values
cnt=0
for i in range(len(pres)):
    if 0.5<pres[i]<0.55:
        pres[i]-=0.05
        cnt+=1
print(cnt)
df['score']=pres
df.to_csv("../submission/submit_8485_guize.csv",index=False)