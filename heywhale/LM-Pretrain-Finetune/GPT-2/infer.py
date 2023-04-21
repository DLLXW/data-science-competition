
"""
Sample from a trained model
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import pandas as pd
# -----------------------------------------------------------------------------
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float32' # 'float32' or 'bfloat16' or 'float16'
#exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
model_args={
            'block_size': 1024, 'n_layer': 12, 'n_head': 12, 'n_embd': 768, 
            'dropout': 0.0, 'bias': False,
            'block_size': 256,
            'vocab_size':1301
}
# model
# init from a model saved in a specific directory
ckpt_path = './out/20230408_finetune/best.pth'
state_dict = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
#print(model)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
# encode the beginning of the prompt

test_df=pd.read_csv('data/diagnosis/preliminary_a_test.csv',header = None)
sub_df=pd.read_csv('data/diagnosis/preliminary_a_sub.csv',header = None)

test_df.columns=['report_ID','description']
sub_df.columns=['report_ID','diagnosis']
test_df

max_new_tokens = 50 # number of tokens generated in each sample
temperature = 0.9 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 5 # retain only the top_k most likely tokens, clamp others to have 0 probability

from tqdm import tqdm
res_col=[]
for start_ids in tqdm(test_df['description'].values):
    start_ids=[int(i) for i in start_ids.split(' ')]
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    #
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            pre = y[0].tolist()
            so=len(start_ids)
            eo=len(pre)
            for i in range(len(pre)-1,0,-1):
                if pre[i]==2:
                    eo=i
                    break
            #
            if so<eo:
                diag = pre[so:eo]
            else:
                diag = start_ids
            diag = [str(i) for i in diag]
            diag=' '.join(diag)
            res_col.append(diag)
            #print(diag)
            #print('---------------')
            #break
#
sub_df['diagnosis']=res_col
sub_df

sub_df.to_csv('./submit/submit_baseline_v2.csv', header=None,index=False)