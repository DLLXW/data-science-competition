from transformers import *
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from model import AlbertClassfier
from tqdm import tqdm
class DataInfer(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return np.array(self.data[index])
if __name__=='__main__':
    # pretrained = 'bert-base-chinese'
    # tokenizer = BertTokenizer.from_pretrained(pretrained)
    # model = BertModel.from_pretrained(pretrained)
    # config = BertConfig.from_pretrained(pretrained)
    # #
    # device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    # albertBertClassifier = AlbertClassfier(model, config, 2)
    # albertBertClassifier.to(device)
    # albertBertClassifier = nn.DataParallel(albertBertClassifier)
    # weights = torch.load('/home/admins/qyl/tianma/huggface_bert/ckpt_group500_labelSmooth/5_542_0_state_dict.pth')
    # albertBertClassifier.load_state_dict(weights)
    # albertBertClassifier.eval()
    # print("model download finished..............")
    # #
    # file_path = "dataset/testA_data_group500.pkl"
    # with open(file_path, 'rb') as fr:
    #     test_df = pickle.load(fr)
    # test_data=test_df['data']
    # sessionIds = test_df['session']
    # test_dataset = DataInfer(test_data)
    # test_dataloader = DataLoader(test_dataset,shuffle=False,batch_size=20)
    # out_list = []
    # for _, token_ids in enumerate(test_dataloader):
    #     token_ids = token_ids.to(device)
    #     with torch.no_grad():
    #         out = albertBertClassifier(token_ids)
    #         out_list += F.softmax(out, dim=1).cpu().numpy()[:, 1].tolist()
    # submit = {'SessionId': sessionIds, 'Probability': out_list}
    # sub_df = pd.DataFrame(submit)
    # sub_df.to_csv('submit/testA_data_group500.csv',index=False)
    # print(sub_df)
    # print(sub_df.shape)
    raw_test=pd.read_csv('/home/admins/qyl/tianma/data/Test_A.csv')['SessionId'].unique()
    sub = {'SessionId': [], 'Probability': []}
    sub_pd=pd.read_csv('/home/admins/qyl/tianma/huggface_bert/submit_group50/epoch_8_831_0.csv')
    pre={}
    for index,name,pro in sub_pd.itertuples():
        pre[name]=pro
    for name in raw_test:
        sub['SessionId'].append(name)
        sub['Probability'].append(pre[name])
    pd.DataFrame(sub).to_csv("yyx_v2.csv")
    print(pd.DataFrame(sub))