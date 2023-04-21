import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
import numpy as np
from transformers import BartForConditionalGeneration
import torch
from tqdm import tqdm
from modeling_cpt import CPTForConditionalGeneration
model = BartForConditionalGeneration.from_pretrained("custom_pretrain_bart/")
#model = BartForConditionalGeneration.from_pretrained("bart-chinese")
#model.resize_token_embeddings(1401)
#
device='cuda'
#ckpt_path = '/home/trojanjet/project/weiqin/diag/bart/output/20230411_bs24_cn/best.pth'
ckpt_path = './output/20230420/epoch_last.pth'  
state_dict = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(state_dict)
print("load_ckpt from .....{}".format(ckpt_path))
model.to(device)
model.eval()
#

test_df=pd.read_csv('./data/diagnosis/preliminary_b_test.csv',header = None)
#sub_df=pd.read_csv('./data/diagnosis/preliminary_a_sub.csv',header = None)
test_df.columns=['report_ID','description']
#sub_df.columns=['report_ID','diagnosis']
#test_df
#
max_length=256
res_col=[]
for desc in tqdm(test_df['description'].values):
    desc=[101]+[int(i)+100 for i in desc.split(' ')]+[102]
    context_len=len(desc)
    #desc=desc+[1]*(max_length-len(desc)) #padding
    #desc_mask=np.array([1]*context_len+[0]*(max_length-context_len)) #attention mask
    desc_mask=np.array([1]*context_len)
    desc_id=torch.from_numpy(np.array([desc]))
    desc_mask=torch.from_numpy(np.array([desc_mask]))
    #print(desc_id.shape,desc_mask.shape)
    #
    with torch.no_grad():
        summaries = model.generate(
            input_ids=desc_id.to(device),
            attention_mask=desc_mask.to(device),
            #decoder_start_token_id=101,
            num_beams=5,
            length_penalty=2.0,
            max_length=80,  # +2 from original because we start at step=1 and stop before max_length
            min_length=15,  # +1 from original because we start at step=1
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=False,
        )  # change these arguments if you want
        #print(summaries)
        #break
        summaries = summaries.cpu().numpy().tolist()[0]
        diag = summaries[1:-1]
        diag = [str(i-100) for i in diag]
        diag=' '.join(diag)
        res_col.append(diag)
        #print(diag)
        #print('---------------')
        #break
#
test_df['diagnosis']=res_col
test_df[['report_ID','diagnosis']].to_csv('./submit_roundb/sub_v1.csv', header=None,index=False)