#
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import transformers
class Bert(nn.Module):
    def __init__(self,bert_name,class_num=13,dropout=0.25):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(bert_name)
        self.bert_out_dim = self.bert.pooler.dense.out_features
        #
        self.txt_head = nn.Sequential(
                    nn.Linear(self.bert_out_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(),
                    #nn.Dropout(p = dropout),
                    #
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(),
                    #nn.Dropout(p = dropout),
                )
        self.img_head=nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(),
                    #nn.Dropout(p = dropout),
                    #
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(),
                    #nn.Dropout(p = dropout),
                    #
                )
        self.classify=nn.Sequential(
                #
                nn.Linear(1024, 512),
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
    def forward(self,text,frt):
        frt_text=self.bert(text)[0]
        frt_text=torch.mean(frt_text,axis=1)#[Batch,768]
        frt_text=self.txt_head(frt_text)
        frt_img=self.img_head(frt)
        frt = torch.cat([frt_text,frt_img],axis=1)
        logit = self.classify(frt)
        return logit
        

def tokenize(txt):
    txt_tok = tokenizer(txt, return_tensors='pt', padding='max_length', truncation=True, max_length=18)
    return txt_tok
if __name__=="__main__":
    bert_name="./chinese_bert/chinese-roberta-wwm-ext"
    tokenizer = transformers.AutoTokenizer.from_pretrained('./chinese_bert/chinese-roberta-wwm-ext')
    texts=['2021年春季微喇裤牛仔裤蓝色常规厚度九分裤女装','2021年春季锥形裤牛仔裤蓝色常规厚度短裤女装']
    tokens = tokenize(texts)
    frt=torch.randn(2,2048).cuda()
    #tokens = {k: v.cuda() for k, v in tokens.items()}
    tokens = tokens['input_ids'].cuda()
    #bert = transformers.AutoModel.from_pretrained('./chinese_bert/chinese-roberta-wwm-ext').cuda()
    #print(tokens.keys())
    #out=bert(**tokens)[0]
    model=Bert(bert_name).cuda()
    out=model(tokens,frt)
    print(out.shape)
