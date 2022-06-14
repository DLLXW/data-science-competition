

import numpy as np
from sklearn import metrics
import os
from sklearn.model_selection import GroupKFold,KFold
from dataset_bert import textDataset
import torch
from torch.utils.data import DataLoader
import time
import random
import shutil
import json
import argparse
import logging
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn 
import torch.nn.functional as F 
from bert import Bert
random.seed(2022)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
#
label_fine_dir='../../data/tmp_data/label_fine.json'
label_coarse_dir='../../data/tmp_data/label_coarse_tuwen.json'
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
class LabelSmoothLoss(nn.Module): 
    def __init__(self, classes=5, smoothing=0.1, dim=-1): 
        super(LabelSmoothLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 

    def forward(self, pred, target): 
        pred = pred.log_softmax(dim=self.dim) 
        with torch.no_grad():
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
def train_model(model,criterion, optimizer, lr_scheduler=None):
    total_iters=len(trainloader)
    since = time.time()
    best_acc = 0
    best_epoch = 0
    #
    warmup_scheduler.step(0)
    iters = len(trainloader)
    for epoch in range(max_epoch):
        model.train(True)
        begin_time=time.time()
        count=0
        train_loss = []
        for i, (tokens, labels, frt) in enumerate(trainloader):
            count+=1
            #tokens = {k: v.cuda() for k, v in tokens.items()}
            #print(tokens)
            tokens=tokens.cuda()
            tokens=torch.squeeze(tokens,dim=1)
            labels = labels.to(device).long()
            frt = frt.to(device).float()
            #
            out_linear= model(tokens,frt)
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
                    ' Fold:{} Epoch:{}({}/{}) loss:{:.3f} lr:{:.8f} epoch_Time:{}min:'.format(
                        fold+1,epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
            #
            train_loss.append(loss.item())
        #lr_scheduler.step()
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
        if label[0]==1:
            if pre[0]==1:
                tuwen_correct_cnt+=1
            tuwen_query_cnt+=1
            #
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
        tokens, labels,frt = data
        tokens=tokens.cuda()
        tokens=torch.squeeze(tokens,dim=1)
        labels = labels.to(device).long()
        labels = labels.type(torch.LongTensor)
        labels, frt = labels.cuda(),frt.cuda().float()
        outputs = model(tokens,frt)
        #pres_list+=outputs.sigmoid().detach().cpu().numpy().tolist()
        pres=F.softmax(outputs,dim=1).detach().cpu().numpy()#.tolist()
        pres=np.argmax(pres,axis=1)
        pres_list+=pres.tolist()
        labels_list+=labels.detach().cpu().numpy().tolist()
    #
    tuwen_acc=metrics.accuracy_score(labels_list, pres_list)
    return tuwen_acc,tuwen_acc,tuwen_acc#tuwen_acc,attr_acc

if __name__=="__main__":
    #chinese-roberta-wwm-ext chinese-macbert-base roberta-base-word-chinese-cluecorpussmall
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_type',type=str, default="roberta",help='roberta/mac/word')
    args = parser.parse_args()
    bert_name_dic={
                'roberta':'../../data/pretrain_model/chinese_bert/chinese-roberta-wwm-ext',
                'mac':'../../data/pretrain_model/chinese_bert/chinese-macbert-base',
                'word':'../../data/pretrain_model/chinese_bert/roberta-base-word-chinese-cluecorpussmall',
                }
    bert_name_length_dic={
                'roberta':50,
                'mac':50,
                'word':30,
                }
    #
    bert_type=args.bert_type
    bert_name=bert_name_dic[bert_type]
    max_length=bert_name_length_dic[bert_type]
    model_save_dir = 'ckpt_offline_'+bert_type+'_tuwen/'
    max_lr=5e-5
    #
    print_interval=100
    train_batch_size=64
    val_batch_size=64
    max_epoch=10
    device = torch.device('cuda')
    criterion = LabelSmoothLoss(2)
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    logger = get_logger(os.path.join(model_save_dir,'log.log'))
    logger.info('bert_type: {} '.format(bert_type))
    logger.info('max_length: {} '.format(max_length))
    logger.info('batch size: {} '.format(train_batch_size))
    logger.info('max_lr: {} '.format(max_lr))
    shutil.copy('./dataset_bert.py',model_save_dir+'/dataset_bert.py')
    shutil.copy('./train_bert_tuwen.py',model_save_dir+'/train_bert_tuwen.py')
    #
    folds = GroupKFold(n_splits=5).split(np.arange(len(groups)),
                    groups=groups)
    kfold_best=[]
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # 
        # if fold in [2,3]:
        #     continue
        logger.info('train fold: {} len train: {} len val: {}'.format(fold+1,len(trn_idx),len(val_idx)))
        model=Bert(bert_name,2)
        model.to(device)
        train_dataset = textDataset(
                label_fine_dir,
                feature_fine_dir,
                label_coarse_dir,
                feature_coarse_dir,
                add_attr=False,
                bert_name=bert_name,
                index=trn_idx,
                mode='train',
                only_tuwen=True,
                max_length=max_length)
        trainloader = DataLoader(train_dataset,
                                batch_size=train_batch_size,
                                shuffle=True,
                                num_workers=8)
        val_dataset = textDataset(
                label_fine_dir,
                feature_fine_dir,
                label_coarse_dir,
                feature_coarse_dir,
                add_attr=False,
                bert_name=bert_name,
                index=val_idx,
                mode='val',
                only_tuwen=True,
                max_length=max_length)
        val_loader = DataLoader(val_dataset,
                                batch_size=val_batch_size,
                                shuffle=False,
                                num_workers=8)
        #
        iter_per_epoch=len(trainloader)
        warmup_epoch=1
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr ,weight_decay=5e-4)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch,start_lr=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max_epoch+1, T_mult=1, eta_min=0.01*max_lr, last_epoch=-1)
        best_loss=train_model(model,criterion, optimizer,lr_scheduler=lr_scheduler)
        kfold_best.append(best_loss)
    logger.info("local cv: {} {}".format(kfold_best,np.mean(kfold_best)))