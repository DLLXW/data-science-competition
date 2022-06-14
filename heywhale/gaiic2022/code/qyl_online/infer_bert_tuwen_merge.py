

import torch
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from bert import Bert
import transformers
import torch.nn.functional as F 
import numpy as np
def load_model(bert_name,weight_path):
    print(weight_path)
    model=Bert(bert_name,2)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict(text_1,text_2,text_3,feature):
    model=model_list_1[0]
    outputs=model(text_1,feature)
    pres=F.softmax(outputs,dim=1).detach().cpu().numpy()[0]
    pres=pres*weight_lst[0]
    cnt=1
    for i in range(1,len(model_list_1)):
        model=model_list_1[i]
        outputs=model(text_1,feature)
        outputs=F.softmax(outputs,dim=1).detach().cpu().numpy()[0]
        pres=pres+weight_lst[cnt]*outputs
        cnt+=1
    #
    for i in range(len(model_list_2)):
        model=model_list_2[i]
        outputs=model(text_2,feature)
        outputs=F.softmax(outputs,dim=1).detach().cpu().numpy()[0]
        pres=pres+weight_lst[cnt]*outputs
        cnt+=1
    for i in range(len(model_list_3)):
        model=model_list_3[i]
        outputs=model(text_3,feature)
        outputs=F.softmax(outputs,dim=1).detach().cpu().numpy()[0]
        pres=pres+weight_lst[cnt]*outputs
        cnt+=1
    print(pres)
    pres_out=np.argmax(pres)
    return pres_out,pres


if __name__=="__main__":
    #
    weight_lst=[]
    weight_lst+=list(0.4*np.array([0.08]*5+[0.15]*4))
    weight_lst+=list(0.35*np.array([0.04]*5+[0.2]*4))
    weight_lst+=list(0.25*np.array([0.14]*5+[0.06]*5))
    print(weight_lst)
    print(np.sum(weight_lst))
    assert abs(np.sum(weight_lst)-1)<0.001
    #../code/chinese_bert/chinese-macbert-base  chinese-roberta-wwm-ext
    bert_name_1='../../data/pretrain_model/chinese_bert/chinese-roberta-wwm-ext'#
    tokenizer_1 = transformers.AutoTokenizer.from_pretrained(bert_name_1)
    bert_name_2='../../data/pretrain_model/chinese_bert/chinese-macbert-base'#
    tokenizer_2 = transformers.AutoTokenizer.from_pretrained(bert_name_2)
    bert_name_3='../../data/pretrain_model/chinese_bert/roberta-base-word-chinese-cluecorpussmall'#
    tokenizer_3 = transformers.AlbertTokenizer.from_pretrained(bert_name_3)
    #outfile
    out_file='submit/submit_0422_merge_tuwen_roundB_v2.txt'
    out_file_prob='submit/submit_0422_merge_tuwen_prob_roundB_v2.txt'
    #
    
    device=torch.device('cuda')
    class_name=['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
    submit_sample={"img_name":"test000255","match":0}
    submit_sample_prob={"img_name":"test000255","match":[]}
    submit_prob=[]
    submit=[]
    model_list_1=[]
    for i in range(5):
        #9334
        model_list_1.append(load_model(bert_name_1,'../../data/model_data/ckpt_0417_roberta_tuwen_fine_coarse/fold_'+str(i+1)+'_best.pth'))
        #9290
        if i<4:
            model_list_1.append(load_model(bert_name_1,'../../data/model_data/ckpt_roberta_tuwen_warmup/fold_'+str(i+1)+'_best.pth'))
    model_list_2=[]
    for i in range(5):
        #9277
        model_list_2.append(load_model(bert_name_2,'../../data/model_data/ckpt_0417_macbert_tuwen_fine_coarse/fold_'+str(i+1)+'_best.pth'))
        #9294
        if i<4:
            model_list_2.append(load_model(bert_name_2,'../../data/model_data/ckpt_macbert_tuwen_warmup/fold_'+str(i+1)+'_best.pth'))
    model_list_3=[]
    for i in range(5):
        #9344
        model_list_3.append(load_model(bert_name_3,'../../data/model_data/ckpt_0418_wordbert_tuwen_fine_coarse/fold_'+str(i+1)+'_best.pth'))
        #9270
        model_list_3.append(load_model(bert_name_3,'../../data/model_data/ckpt_wordbert_tuwen_warmup/fold_'+str(i+1)+'_best.pth'))
    #
    assert len(weight_lst)==len(model_list_1+model_list_2+model_list_3)
    #
    Threshold=0.5
    data_dir='../data/preliminary_testB.txt'
    with open(data_dir, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            img_name=item['img_name']
            title=item['title']
            query=item['query']
            feature=torch.tensor(item['feature'])
            feature=feature.unsqueeze(0)
            feature=feature.cuda().float()
            text_1=tokenizer_1(title, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
            text_1=text_1['input_ids']
            text_1=torch.tensor(text_1).cuda()
            # #
            text_2=tokenizer_2(title, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
            text_2=text_2['input_ids']
            text_2=torch.tensor(text_2).cuda()
            #
            text_3=tokenizer_3(title, return_tensors='pt', padding='max_length', truncation=True, max_length=30)
            text_3=text_3['input_ids']
            text_3=torch.tensor(text_3).cuda()
            pre,pres_prob=predict(text_1,text_2,text_3,feature)
            #pre=predict(text_1,None,None,feature)
            submit_sample["img_name"]=img_name
            submit_sample["match"]=int(pre)
            #
            submit_sample_prob["img_name"]=img_name
            submit_sample_prob["match"]=[float(pres_prob[0]),float(pres_prob[1])]
            #print(submit_sample)
            submit.append(json.dumps(submit_sample, ensure_ascii=False)+'\n')
            submit_prob.append(json.dumps(submit_sample_prob, ensure_ascii=False)+'\n')
    #
    with open(out_file, 'w') as f:
        f.writelines(submit)
    with open(out_file_prob, 'w') as f:
        f.writelines(submit_prob)