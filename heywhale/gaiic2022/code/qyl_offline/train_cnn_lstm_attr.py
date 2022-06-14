
#
import numpy as np
import shutil
from sklearn import metrics
import os
import argparse
import logging
from sklearn.model_selection import GroupKFold,KFold
from net_CNN import CNN_Text
from net_BiLSTM import LSTM_Text
from dataset_qyl import textDataset
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import time
import warnings
warnings.filterwarnings("ignore")
import random
import json
import jieba
jieba.setLogLevel(jieba.logging.INFO)
import logging

random.seed(2022)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
label_fine_dir='../../data/tmp_data/label_fine.json'
label_coarse_dir='../../data/tmp_data/label_coarse_attr.json'#根据coarse 的 title 提取的关键属性
feature_fine_dir='../../data/tmp_data/feature_imgName_fine.json'
feature_coarse_dir='../../data/tmp_data/feature_imgName_coarse.json'

with open(label_fine_dir, 'r') as f:
    groups = json.load(f)['img_name']
# 
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self,  optimizer, total_iters, start_lr=1e-6,last_epoch=-1):
        self.total_iters = total_iters
        self.start_lr=start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [self.start_lr+base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
def get_vocab(input_path):
    if os.path.exists(input_path):
        with open(input_path, 'r') as f:
            word_to_idx = json.load(f)
    else:
        with open(label_fine_dir, 'r') as f:
            lable_fine_title = json.load(f)
        titles_fine_all=lable_fine_title['title']
        with open(label_coarse_dir, 'r') as f:
            lable_coarse_title = json.load(f)
        titles_coarse_all=lable_coarse_title['title']
        #构建词汇表
        #cnt=[]
        vocab=[]
        for x in titles_fine_all:
            vocab+=[w for w in jieba.cut(x)]
        for x in titles_coarse_all:
            vocab+=[w for w in jieba.cut(x)]
        #print(np.mean(cnt),np.median(cnt),np.max(cnt),np.min(cnt))
        vocab=list(set(vocab))
        word_to_idx = {word:idx+1 for idx,word in enumerate(vocab)}
        word_to_idx['<unk>'] = 0
        idx_to_word = {idx+1:word for idx,word in enumerate(vocab)}
        idx_to_word[0] = '<unk>'
        with open(input_path,"w") as f:
            json.dump(word_to_idx,f)
    return word_to_idx
#
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
#
def train_model(model,criterion, optimizer, lr_scheduler=None):
    total_iters=len(trainloader)
    #print('total_iters:{}'.format(total_iters))
    since = time.time()
    best_acc = 0
    best_epoch = 0
    #
    iters = len(trainloader)
    for epoch in range(max_epoch):
        model.train(True)
        begin_time=time.time()
        # print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        # print('Fold{} Epoch {}/{}'.format(fold+1,epoch, max_epoch))
        # print('-' * 10)
        count=0
        train_loss = []
        for i, (inputs, labels, frt) in enumerate(trainloader):
            count+=1
            inputs = inputs.type(torch.LongTensor).to(device)
            labels = labels.to(device).float()
            frt = frt.to(device).float()
            #
            out_linear= model(inputs,frt)
            loss = criterion(out_linear, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新cosine学习率
            if lr_scheduler!=None:
                if epoch >= warmup_epoch:
                    lr_scheduler.step(epoch + count / iters)
                else:
                    warmup_scheduler.step()
            if print_interval>0 and (i % print_interval == 0 or out_linear.size()[0] < train_batch_size):
                spend_time = time.time() - begin_time
                logger.info(
                    ' Fold:{} Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        fold+1,epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
            #
            train_loss.append(loss.item())
        val_auc,tuwen_acc,attr_acc= val_model(model)
        val_acc=0.5*tuwen_acc+0.5*attr_acc
        logger.info('val auc: {:.4f} val acc: {:.4f} tuwen_acc: {:.4f} attr_acc: {:.4f}'.format(val_auc,val_acc,tuwen_acc,attr_acc))
        model_out_path = model_save_dir+"/"+'fold_'+str(fold+1)+'_'+str(epoch) + '.pth'
        best_model_out_path = model_save_dir+"/"+'fold_'+str(fold+1)+'_best'+'.pth'
        #save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch=epoch
            torch.save(model.state_dict(), best_model_out_path)
            logger.info("save best epoch: {} best auc: {} best acc: {}".format(best_epoch,val_auc,best_acc))
        #save based on epoch interval
        #if epoch % 5  == 0 and epoch>30:
            #torch.save(model.state_dict(), model_out_path)
    #
    logger.info('Fold{} best_acc: {:.3f} Best epoch:{}'.format(fold+1,best_acc,best_epoch))
    time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return best_acc
def cal_acc(labels_list, pres_list):
    '''
    [val_len,13] 
    '''
    tuwen_correct_cnt=0
    attr_correct_cnt=0
    query_cnt=0
    tuwen_query_cnt=0
    for i in range(len(labels_list)):
        label=np.array(labels_list[i])
        pre=np.array(pres_list[i])
        pre=(pre>0.5).astype(int)
        #if label[0]==1:
        if pre[0]==label[0]:
            tuwen_correct_cnt+=1
        tuwen_query_cnt+=1
        #
        if label[0]==1:
            query_cnt+=np.sum(label[1:])
            for j in range(1,len(label)):
                if label[j]==1 and pre[j]==1:
                    attr_correct_cnt+=1
    tuwen_acc=tuwen_correct_cnt/tuwen_query_cnt
    attr_acc=attr_correct_cnt/query_cnt
    return tuwen_acc,attr_acc

@torch.no_grad()
def val_model(model):
    model.eval()
    pres_list=[]
    labels_list=[]
    for data in val_loader:
        inputs, labels,frt = data
        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        inputs, labels, frt = inputs.cuda(), labels.cuda(),frt.cuda().float()
        outputs = model(inputs,frt)
        #pres_list.append(outputs.sigmoid().detach().cpu().numpy())
        #labels_list.append(labels.detach().cpu().numpy())
        pres_list+=outputs.sigmoid().detach().cpu().numpy().tolist()
        labels_list+=labels.detach().cpu().numpy().tolist()
    #
    val_auc = metrics.roc_auc_score(labels_list, pres_list, multi_class='ovo')
    tuwen_acc,attr_acc=cal_acc(labels_list, pres_list)
    return val_auc,tuwen_acc,attr_acc
#
if __name__ == "__main__":
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str, default="lstm",help='lstm/cnn')
    args = parser.parse_args()
    model_name=args.model_name#cnn lstm
    #
    word_to_idx=get_vocab('../../data/tmp_data/word_to_idx_fine_coarse_attr.json')
    model_save_dir ='ckpt_offline_'+model_name+'_attr/'
    print_interval=100
    max_lr=5e-4
    train_batch_size=64
    val_batch_size=64
    max_epoch=8
    embed_num=len(word_to_idx)+1
    device = torch.device('cuda')
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    logger = get_logger(os.path.join(model_save_dir,'log.log'))
    logger.info("embed_num: {}".format(embed_num))
    logger.info('model_name: {} '.format(model_name))
    logger.info('batch size: {} '.format(train_batch_size))
    logger.info('max_lr: {} '.format(max_lr))
    shutil.copy('./dataset_bert.py',model_save_dir+'/dataset_bert.py')
    shutil.copy('./train_bert_tuwen.py',model_save_dir+'/train_bert_tuwen.py')
    
    folds = GroupKFold(n_splits=5).split(np.arange(len(groups)),
                    groups=groups)
    kfold_best=[]
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # 
        # if fold<3:
        #     continue
        logger.info('train fold: {} len train: {} len val: {}'.format(fold+1,len(trn_idx),len(val_idx)))
        if model_name=='cnn':
            model=CNN_Text(embed_num,class_num=13)
        else:
            model=LSTM_Text(embed_num,class_num=13)
        model.to(device)
        #optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4 ,weight_decay=5e-4)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=1, eta_min=5e-6, last_epoch=-1)
        train_dataset = textDataset(
                label_fine_dir,
                feature_fine_dir,
                label_coarse_dir,
                feature_coarse_dir,
                add_attr=False,
                word_to_idx=word_to_idx,
                index=trn_idx,
                mode='train',
                only_tuwen=False)
        trainloader = DataLoader(train_dataset,
                                batch_size=train_batch_size,
                                shuffle=True,
                                num_workers=8)
        val_dataset = textDataset(label_fine_dir,
                        feature_fine_dir,
                        label_coarse_dir,
                        feature_coarse_dir,
                        add_attr=False,
                        word_to_idx=word_to_idx,
                        index=val_idx,
                        mode='val',
                        only_tuwen=False)
        val_loader = DataLoader(val_dataset,
                                batch_size=val_batch_size,
                                shuffle=False,
                                num_workers=8)
        iter_per_epoch=len(trainloader)
        warmup_epoch=1
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr ,weight_decay=5e-4)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch,start_lr=5e-6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max_epoch+1, T_mult=1, eta_min=0.01*max_lr, last_epoch=-1)
        best_loss=train_model(model,criterion, optimizer,lr_scheduler=lr_scheduler)
        kfold_best.append(best_loss)
    logger.info("local cv: {} kfold_best: {}",kfold_best,np.mean(kfold_best))