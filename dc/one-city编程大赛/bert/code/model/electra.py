import torch
class ElectraClassfier(torch.nn.Module):
    def __init__(self,bert_model,bert_config,num_class):
        super(ElectraClassfier,self).__init__()
        self.bert_model=bert_model
        self.dropout=torch.nn.Dropout(0.4)
        self.fc1=torch.nn.Linear(bert_config.hidden_size,bert_config.hidden_size)
        self.fc2=torch.nn.Linear(bert_config.hidden_size,num_class)
    def forward(self,token_ids):
        bert_out=self.bert_model(token_ids)[0].mean(1) #句向量 [batch_size,hidden_size]
        bert_out=self.dropout(bert_out)
        bert_out=self.fc1(bert_out)
        bert_out=self.dropout(bert_out)
        bert_out=self.fc2(bert_out) #[batch_size,num_class]
        return bert_out
#
