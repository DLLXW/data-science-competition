

from net import CNN_Text
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
def load_model(weight_path):
    print(weight_path)
    model=CNN_Text(embed_num=859)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict(texts):
    pres_all=[]
    for text in tqdm(texts):
        text=[int(i) for i in text.split(' ')]
        if len(text)>50:
            text=text[:50]
        else:
            text=text+[858]*(50-len(text))
        #
        text=torch.from_numpy(np.array(text))
        text=text.unsqueeze(0)
        text=text.type(torch.LongTensor).cuda()
        #
        for i in range(len(model_list)):
            model=model_list[i]
            outputs=model(text)
            outputs=outputs.sigmoid().detach().cpu().numpy()[0]
            if i==0:
                pres_fold=outputs/len(model_list)
            else:
                pres_fold+=outputs/len(model_list)
        #
        pres_fold=[str(p) for p in pres_fold]
        pres_fold=' '.join(pres_fold)
        pres_all.append(pres_fold)
    return pres_all

if __name__=="__main__":
    device=torch.device('cuda')
    model_list=[]
    for i in range(5):
        model_list.append(load_model('ckpt/fold_'+str(i+1)+'_best.pth'))
    #
    test_df=pd.read_csv('data/track1_round1_testA_20210222.csv',header=None) 
    #
    test_df.columns=['report_ID','description']
    submit=test_df.copy()
    print("test_df:{}".format(test_df.shape))
    new_des=[i.strip('|').strip() for i in test_df['description'].values]
    test_df['description']=new_des
    sub_id=test_df['report_ID'].values
    #
    print(sub_id[0])
    save_dir='submits/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    pres_all=predict(new_des)
    str_w=''
    with open(save_dir+'submit.csv','w') as f:
        for i in range(len(sub_id)):
            str_w+=sub_id[i]+','+'|'+pres_all[i]+'\n'
        str_w=str_w.strip('\n')
        f.write(str_w)
    #