from transformers import *
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import time
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
import os
import pickle
from model import AlbertClassfier
import pandas as pd
from tqdm import tqdm
from pytorch_toolbelt import losses as L
import random
from utils import LabelSmoothSoftmaxCE
random.seed(2020)
#
class DataGen(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return np.array(self.data[index]), np.array(self.label[index])
#
class DataInfer(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return np.array(self.data[index])
#
if __name__=='__main__':
    decode_dic={0: '文化休闲',
                 1: '医疗卫生',
                 2: '经济管理',
                 3: '教育科技',
                 4: '城乡建设',
                 5: '工业',
                 6: '民政社区',
                 7: '交通运输',
                 8: '生态环境',
                 9: '政法监察',
                 10: '农业畜牧业',
                 11: '文秘行政',
                 12: '劳动人事',
                 13: '资源能源',
                 14: '信息产业',
                 15: '旅游服务',
                 16: '商业贸易',
                 17: '气象水文测绘地震地理',
                 18: '财税金融',
                 19: '外交外事'}
    #使用'voidful/albert_chinese_tiny'预训练
    #pretrained = 'voidful/albert_chinese_small'
    pretrained = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model = BertModel.from_pretrained(pretrained)
    config = BertConfig.from_pretrained(pretrained)
    #
    albertBertClassifier = AlbertClassfier(model, config, 20)
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    albertBertClassifier = albertBertClassifier.to(device)
    #
    max_num_epochs = 21
    val_size=0.01
    print_interval=50
    save_interval=2
    sumit_dir="submit_title/"
    save_ckpt='ckpt_title/'
    if not os.path.exists(save_ckpt):
        os.mkdir(save_ckpt)
    if not os.path.exists(sumit_dir):
        os.mkdir(sumit_dir)
    n_gpu=2
    if n_gpu>1:
        albertBertClassifier=torch.nn.DataParallel(albertBertClassifier)
    #
    #weights_class = torch.FloatTensor([0.1, 0.9])
    criterion = torch.nn.CrossEntropyLoss()
    use_multi_loss=True
    if use_multi_loss:
        criterion = L.JointLoss(first=torch.nn.CrossEntropyLoss().cuda(), second=LabelSmoothSoftmaxCE().cuda(),
                                first_weight=0.5, second_weight=0.5)
    #optimizer = torch.optim.SGD(albertBertClassifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    # warm_up_with_multistep_lr
    optimizer = AdamW(albertBertClassifier.parameters(),lr= 2e-5)
    #
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, last_epoch=-1)
    print('download model finished ...............')
    #
    file_path = "usr_data/train_data.pkl"
    with open(file_path, 'rb') as fr:
        train_data = pickle.load(fr)
    X_train, X_test, y_train, y_test = train_test_split(train_data['data'], train_data['label'], test_size=val_size, shuffle=True,random_state=2020)
    print(len(X_train), len(X_test), len(y_train), len(y_test), len(X_train[0]))
    print('download raw-data finished ...........')
    #
    train_dataset = DataGen(X_train, y_train)
    test_dataset = DataGen(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, shuffle=True,batch_size=128)
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    #
    # inference dataset
    file_path_infer = "usr_data/testA_data.pkl"
    with open(file_path_infer, 'rb') as fr:
        infer_df = pickle.load(fr)
    infer_data = infer_df['data']
    sessionIds = infer_df['name']
    infer_dataset = DataInfer(infer_data)
    infer_dataloader = DataLoader(infer_dataset, shuffle=False, batch_size=128)
    #
    total_iters=len(train_dataloader)
    for epoch in range(1,max_num_epochs):
        loss_sum = 0.0
        accu = 0
        albertBertClassifier.train()
        begin_time=time.time()
        for step, (token_ids, label) in enumerate(train_dataloader):
            token_ids = token_ids.to(device)
            label = label.to(device)
            out = albertBertClassifier(token_ids)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
            loss_sum += loss.cpu().data.numpy()
            accu += (out.argmax(1) == label).sum().cpu().data.numpy()
            if step%print_interval==0 and step!=0:
                spend_time=time.time()-begin_time
                print("epoch{},[{}/{}],train_loss:{},ETA:{}min".format(epoch,step,total_iters,loss,spend_time/ step * total_iters // 60 - spend_time // 60))
        #
        scheduler.step()
        test_loss_sum = 0.0
        test_acc = 0
        albertBertClassifier.eval()
        out_all=[]
        out_pre=[]
        label_all=[]
        for step, (token_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.to(device)
            label = label.to(device)
            with torch.no_grad():
                out = albertBertClassifier(token_ids)
                loss = criterion(out, label)
                test_loss_sum += loss.cpu().data.numpy()
                #out_all+=out.cpu().numpy()[:,1].tolist()
                out_pre+=torch.max(out.data,1)[1].cpu().numpy().tolist()
                label_all+=label.cpu().numpy().tolist()
                test_acc += (out.argmax(1) == label).sum().cpu().data.numpy()
        #test_auc=roc_auc_score(label_all,out_all)
        #f1_val=f1_score(label_all,out_pre)
        acc_val=accuracy_score(label_all,out_pre)
        print(test_acc)
        print("epoch{},train_loss:{},val_loss:{},acc_val:{}".format (
        epoch, loss_sum / len(train_dataloader), test_loss_sum / len(test_dataloader),acc_val))

        if epoch%save_interval==0:
            torch.save(albertBertClassifier.state_dict(), os.path.join(save_ckpt, str(epoch) + '_' + str(int(acc_val*1000))+ '_state_dict.pth'))

        #inference
        print("infer .....")
        out_list = []
        for _, token_ids in enumerate(infer_dataloader):
            token_ids = token_ids.to(device)
            with torch.no_grad():
                out = albertBertClassifier(token_ids)
                out_list += torch.max(out.data,1)[1].cpu().numpy().tolist()

        out_list = [decode_dic[i] for i in out_list]
        submit = {'filename': sessionIds, 'label': out_list}
        sub_df = pd.DataFrame(submit)
        sub_df.to_csv(os.path.join(sumit_dir,'epoch_'+str(epoch) + '_' + str(int(acc_val*1000))+'.csv'), index=False)
        #
