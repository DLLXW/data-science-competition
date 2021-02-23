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
from sklearn.model_selection import StratifiedKFold
from utils.log import get_logger
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
def get_list_list(list_1,list_2):
    return [list_1[i] for i in list_2]
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
    #
    init_lr=2e-5
    max_num_epochs = 15
    val_size=0.01
    print_interval=50
    save_interval=2
    use_optm = 'adm'
    save_ckpt = 'ckpt_kfold_title/'
    submit_dir = 'submit_kfold_title/'
    use_optm = 'AdamW'
    use_multi_loss = True
    n_gpu = 2
    init_lr = 1e-5
    if not os.path.exists(save_ckpt):
        os.mkdir(save_ckpt)
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)
    logger = get_logger(os.path.join(save_ckpt, 'log.log'))
    logger.info('Using pretrained: {}'.format(pretrained))
    logger.info('use_multi_loss: {}'.format(use_multi_loss))
    logger.info('INIT_LR: {}'.format(init_lr))
    #
    file_path = "dataset/train_data.pkl"
    with open(file_path, 'rb') as fr:
        train_data = pickle.load(fr)
    train_all = train_data['data']
    kind = train_data['label']
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20)
    fold = 0
    for train_idx, test_idx in sk.split(train_all, kind):
        # 每一个fold都重新初始化一个新的模型
        #
        #weights_class = torch.FloatTensor([0.1, 0.9])
        # criterion = torch.nn.CrossEntropyLoss(weight=weights_class)
        if use_multi_loss:
            criterion = L.JointLoss(first=torch.nn.CrossEntropyLoss().cuda(), second=LabelSmoothSoftmaxCE().cuda(),
                                    first_weight=0.5, second_weight=0.5)
        #
        tokenizer = BertTokenizer.from_pretrained(pretrained)
        model = BertModel.from_pretrained(pretrained)
        config = BertConfig.from_pretrained(pretrained)

        albertBertClassifier = AlbertClassfier(model, config, 20)
        device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
        albertBertClassifier = albertBertClassifier.to(device)
        if n_gpu > 1:
            albertBertClassifier = torch.nn.DataParallel(albertBertClassifier)
        if use_optm == 'sgd':
            optimizer = torch.optim.SGD(albertBertClassifier.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = AdamW(albertBertClassifier.parameters(), lr=init_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, last_epoch=-1)
        fold += 1
        logger.info('download fold_{} model finished ...............'.format(fold))
        save_ckpt_fold = save_ckpt + 'fold' + str(fold)
        submit_dir_fold = submit_dir + 'fold' + str(fold)
        if not os.path.exists(save_ckpt_fold):
            os.mkdir(save_ckpt_fold)
        if not os.path.exists(submit_dir_fold):
            os.mkdir(submit_dir_fold)
        X_train = get_list_list(train_all, train_idx)
        y_train = get_list_list(kind, train_idx)
        X_test = get_list_list(train_all, test_idx)
        y_test = get_list_list(kind, test_idx)
        #
        train_dataset = DataGen(X_train, y_train)
        test_dataset = DataGen(X_test, y_test)
        train_dataloader = DataLoader(train_dataset, shuffle=True,batch_size=128)
        test_dataloader = DataLoader(test_dataset, batch_size=128)
        #
        # inference dataset
        file_path_infer = "dataset/testA_data.pkl"
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
                torch.save(albertBertClassifier.state_dict(), os.path.join(save_ckpt_fold, str(epoch) + '_' + str(int(acc_val*1000))+ '_state_dict.pth'))

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
            sub_df.to_csv(os.path.join(submit_dir_fold,'epoch_'+str(epoch) + '_' + str(int(acc_val*1000))+'.csv'), index=False)
            #
