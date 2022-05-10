from transformers import *
import torch
import torch.nn as nn
import numpy as np
import torch
class AlbertClassfier(torch.nn.Module):
    def __init__(self,bert_model,bert_config,num_class):
        super(AlbertClassfier,self).__init__()
        self.bert_model=bert_model
        self.dropout=torch.nn.Dropout(0.4)
        self.fc1=torch.nn.Linear(bert_config.hidden_size,bert_config.hidden_size)
        self.fc2=torch.nn.Linear(bert_config.hidden_size,num_class)
    def forward(self,token_ids):
        bert_out=self.bert_model(token_ids)[1] #句向量 [batch_size,hidden_size]
        bert_out=self.dropout(bert_out)
        bert_out=self.fc1(bert_out)
        bert_out=self.dropout(bert_out)
        bert_out=self.fc2(bert_out) #[batch_size,num_class]
        return bert_out
if __name__=='__main__':

    #使用'voidful/albert_chinese_tiny'预训练
    #pretrained = 'voidful/albert_chinese_small'
    pretrained = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model = BertModel.from_pretrained(pretrained)
    config = BertConfig.from_pretrained(pretrained)
    #
    text='我今天去类银行，发现没人'
    token_ids=ids = tokenizer.encode(text.strip(), max_length=20, padding='max_length', truncation=True)
    token_ids=torch.from_numpy(np.array(token_ids)).unsqueeze(0)
    print(token_ids)
    out=model(token_ids)  # 句向量 [batch_size,hidden_size]
    #print(out[0].shape,out[1].shape) #torch.Size([1, 20, 768]) torch.Size([1, 768])
    ct=0
    #print(model)
    print(model.encoder)
    for child in model.encoder.children():
        print(child)
