
import pandas as pd
from pandas import DataFrame
import numpy as np
#from gensim.models import word2vec
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import GroupKFold,KFold
from net import CNN_Text
from dataset import textDataset
import torch
from torch.utils.data import DataLoader
import time
import warnings
warnings.filterwarnings("ignore")
import random
import json
import jieba
random.seed(2022)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
label_dir='../data/label_v1.json'
feature_dir='../data/feature_imgName.json'
with open(label_dir, 'r') as f:
    groups = json.load(f)['img_name']
#
def get_vocab(input_path):
    if os.path.exists(input_path):
        with open(input_path, 'r') as f:
            word_to_idx = json.load(f)
    else:
        with open(label_dir, 'r') as f:
            lable_title = json.load(f)
        titles_all=lable_title['title']
        #构建词汇表
        #cnt=[]
        vocab=[]
        for x in titles_all:
            vocab+=[w for w in jieba.cut(x)]
            #cnt.append(len([w for w in jieba.cut(x)]))
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
def train_model(model,criterion, optimizer, lr_scheduler=None):
    total_iters=len(trainloader)
    print('total_iters:{}'.format(total_iters))
    since = time.time()
    best_loss = 1e7
    best_epoch = 0
    #
    iters = len(trainloader)
    for epoch in range(1,max_epoch+1):
        model.train(True)
        begin_time=time.time()
        print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        print('Fold{} Epoch {}/{}'.format(fold+1,epoch, max_epoch))
        print('-' * 10)
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
                lr_scheduler.step(epoch + count / iters)
            if print_interval>0 and (i % print_interval == 0 or out_linear.size()[0] < train_batch_size):
                spend_time = time.time() - begin_time
                print(
                    ' Fold:{} Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        fold+1,epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
            #
            train_loss.append(loss.item())
        #lr_scheduler.step()
        val_auc,val_loss= val_model(model, criterion)
        print('valLogLoss: {:.4f} valAuc: {:.4f}'.format(val_loss,val_auc))
        model_out_path = model_save_dir+"/"+'fold_'+str(fold+1)+'_'+str(epoch) + '.pth'
        best_model_out_path = model_save_dir+"/"+'fold_'+str(fold+1)+'_best'+'.pth'
        #save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch=epoch
            torch.save(model.state_dict(), best_model_out_path)
            print("save best epoch: {} best auc: {} best logloss: {}".format(best_epoch,val_auc,val_loss))
        #save based on epoch interval
        #if epoch % 5  == 0 and epoch>30:
            #torch.save(model.state_dict(), model_out_path)
    #
    print('Fold{} Best logloss: {:.3f} Best epoch:{}'.format(fold+1,best_loss,best_epoch))
    time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return best_loss

@torch.no_grad()
def val_model(model, criterion):
    dset_sizes=len(val_dataset)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
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
    #preds = np.concatenate(pres_list)
    #labels = np.concatenate(labels_list)
    #val_auc = metrics.roc_auc_score(labels, preds, multi_class='ovo')
    val_auc = metrics.roc_auc_score(labels_list, pres_list, multi_class='ovo')
    log_loss=metrics.log_loss(labels_list, pres_list)#
    return val_auc,log_loss
#
if __name__ == "__main__":
    #
    word_to_idx=get_vocab('../data/word_to_idx_v1.json')
    model_save_dir ='ckpt_v1/'
    print_interval=100
    train_batch_size=64
    val_batch_size=64
    max_epoch=14
    embed_num=len(word_to_idx)+1
    print("embed_num:",len(word_to_idx)+1)
    device = torch.device('cuda')
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    folds = GroupKFold(n_splits=5).split(np.arange(len(groups)),
                    groups=groups)
    kfold_best=[]
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # 
        print('train fold: {} len train: {} len val: {}'.format(fold+1,len(trn_idx),len(val_idx)))
        model=CNN_Text(embed_num,class_num=13)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4 ,weight_decay=5e-4)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=5e-6, last_epoch=-1)
        train_dataset = textDataset(label_dir,feature_dir,word_to_idx,trn_idx)
        trainloader = DataLoader(train_dataset,
                                batch_size=train_batch_size,
                                shuffle=True,
                                num_workers=8)
        val_dataset = textDataset(label_dir,feature_dir,word_to_idx,val_idx)
        val_loader = DataLoader(val_dataset,
                                batch_size=val_batch_size,
                                shuffle=False,
                                num_workers=8)
        best_loss=train_model(model,criterion, optimizer,lr_scheduler=lr_scheduler)
        kfold_best.append(best_loss)
    print("local cv:",kfold_best,np.mean(kfold_best))