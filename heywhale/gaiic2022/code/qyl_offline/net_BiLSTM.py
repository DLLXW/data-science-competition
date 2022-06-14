
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
class LSTM_Text(nn.Module):
    
    def __init__(self, embed_num,class_num=13,dropout=0.25):
        super(LSTM_Text, self).__init__()
        embed_dim=128
        hidden=[128,256,128]
        self.embed = nn.Embedding(embed_num, embed_dim)#词嵌入
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
        self.lstm1 =nn.LSTM(128, hidden[0],
                            batch_first=True, bidirectional=True,dropout=dropout)
        self.lstm2=nn.LSTM(2 * hidden[0], hidden[1],
                            batch_first=True, bidirectional=True,dropout=dropout)
        self.lstm3=nn.LSTM(2 * hidden[1], hidden[2],
                            batch_first=True, bidirectional=True,dropout=dropout)
        self.head=nn.Sequential(
            nn.Linear(2 * hidden[2], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2)
            )
        self.dropout = nn.Dropout(0.5)
    #
    def attention_net(self, x, query, mask=None):

        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim = -1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn
    def forward(self, x, frt):
        x = self.embed(x)  # (N, W, D)-batch,单词数量，维度
        x,_=self.lstm1(x)
        x,_=self.lstm2(x)
        x,_=self.lstm3(x)
        query = self.dropout(x)
        x, _ = self.attention_net(x, query)
        #
        x_img = self.img_head(frt)
        x = torch.cat([x,x_img],axis=1)
        #
        logit = self.classify(x)  # (N, C)
        return logit
if __name__=="__main__":
    net=LSTM_Text(embed_num=1000)
    x=torch.LongTensor([[1,2,4,5,2,35,43,113,111,451,455,22,45,55],[14,3,12,9,13,4,51,45,53,17,57,954,156,23]])
    frt=torch.randn(2,2048)
    logit=net(x,frt)
    print(logit.shape)