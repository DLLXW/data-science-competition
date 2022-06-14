
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import transformers
class LSTM_Text(nn.Module):
    
    def __init__(self, bert_name,class_num=13,dropout=0.25):
        super(LSTM_Text, self).__init__()
        hidden=[512,512,256,128]
        self.bert = transformers.AutoModel.from_pretrained(bert_name,output_hidden_states=True)
        #
        self.img_head=nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(),
                    #nn.Dropout(p = dropout),
                    #
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    #nn.Dropout(p = dropout),
                    #
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Dropout(p = dropout),
                    #
                )
        self.classify=nn.Sequential(
                #
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Dropout(p = dropout),
                #
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(p = dropout),
                nn.Linear(256, class_num)
        )
        #
        self.lstm1 =nn.LSTM(768, hidden[0],
                            batch_first=True, bidirectional=True,dropout=dropout)
        self.lstm2=nn.LSTM(2 * hidden[0], hidden[1],
                            batch_first=True, bidirectional=True,dropout=dropout)
        self.lstm3=nn.LSTM(2 * hidden[1], hidden[2],
                            batch_first=True, bidirectional=True,dropout=dropout)
        self.lstm4=nn.LSTM(2 * hidden[2], hidden[3],
                            batch_first=True, bidirectional=True,dropout=dropout)
        # self.head=nn.Sequential(
        #     nn.Linear(2 * hidden[2], 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 2)
        #     )
        self.dropout = nn.Dropout(0.5)
    #
    def attention_net(self, x, query, mask=None):

        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim = -1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn
    def forward(self, text, frt):
        x=self.bert(text)[2][0]#[Batch,Seq_len,768]
        x,_=self.lstm1(x)
        x,_=self.lstm2(x)
        x,_=self.lstm3(x)
        x,_=self.lstm4(x)
        query = self.dropout(x)
        x, _ = self.attention_net(x, query)
        #
        x_img = self.img_head(frt)
        x = torch.cat([x,x_img],axis=1)
        #
        logit = self.classify(x)  # (N, C)
        return logit
if __name__=="__main__":
    bert_name="../code/chinese_bert/chinese-roberta-wwm-ext"
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_name)
    title=['2021年春季微喇裤牛仔裤蓝色常规厚度九分裤女装','2021年春季锥形裤牛仔裤蓝色常规厚度短裤女装']
    tokens = tokenizer(title, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
    frt=torch.randn(2,2048).cuda()
    #tokens = {k: v.cuda() for k, v in tokens.items()}
    tokens = tokens['input_ids'].cuda()
    net=LSTM_Text(bert_name).cuda()
    #x=torch.LongTensor([[1,2,4,5,2,35,43,113,111,451,455,22,45,55],[14,3,12,9,13,4,51,45,53,17,57,954,156,23]])
    logit=net(tokens,frt)
    print(logit.shape)