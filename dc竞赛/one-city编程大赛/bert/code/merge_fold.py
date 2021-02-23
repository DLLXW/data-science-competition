import pandas as pd

sub_1=pd.read_csv("submit_kfold_title/fold1/epoch_8_968.csv")
sub_2=pd.read_csv("submit_kfold_title/fold2/epoch_8_971.csv")
sub_3=pd.read_csv("submit_kfold_title/fold3/epoch_10_968.csv")
sub_4=pd.read_csv("submit_kfold_title/fold2/epoch_10_972.csv")
sub_5=pd.read_csv("submit_kfold_title/fold1/epoch_10_968.csv")

sub_value_1=sub_1['label'].values
sub_value_2=sub_2['label'].values
sub_value_3=sub_3['label'].values
sub_value_4=sub_4['label'].values
sub_value_5=sub_5['label'].values
cls=sub_1['label'].unique()
most_list=[]
for i in range(len(sub_value_1)):
    vote=[sub_value_1[i],sub_value_2[i],sub_value_3[i],sub_value_4[i],sub_value_5[i]]
    tmp={}
    for k in cls:
        tmp[k]=0
    for j in vote:
        tmp[j]+=1
    most = sorted(tmp.items(), key=lambda item: item[1])[-1][0]
    most_list.append(most)
sub_1['label']=most_list
sub_1.to_csv("vote.csv",index=False)
print(sub_1)