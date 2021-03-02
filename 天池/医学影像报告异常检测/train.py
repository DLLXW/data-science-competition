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
from sklearn.model_selection import StratifiedKFold
from net import CNN_Text
from datasets import textDataset
import torch
from torch.utils.data import DataLoader
import time
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(2021)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
        running_corrects_linear = 0
        count=0
        train_loss = []
        for i, (inputs, labels) in enumerate(trainloader):
            count+=1
            inputs = inputs.type(torch.LongTensor).to(device)
            labels = labels.to(device).float()
            #
            out_linear= model(inputs)
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
        inputs, labels = data
        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
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
    train_df=pd.read_csv('data/track1_round1_train_20210222.csv',header=None)
    test_df=pd.read_csv('data/track1_round1_testA_20210222.csv',header=None) 
    #
    train_df.columns=['report_ID','description','label']
    test_df.columns=['report_ID','description']
    train_df.drop(['report_ID'],axis=1,inplace=True)
    test_df.drop(['report_ID'],axis=1,inplace=True)
    print("train_df:{},test_df:{}".format(train_df.shape,test_df.shape))
    #
    new_des=[i.strip('|').strip() for i in train_df['description'].values]
    new_label=[i.strip('|').strip() for i in train_df['label'].values]
    train_df['description']=new_des
    train_df['label']=new_label
    new_des=[i.strip('|').strip() for i in test_df['description'].values]
    test_df['description']=new_des
    #
    #总共1w条训练数据里面包含2622条正常样本，这里的正常样本标签就使用[0,0,0,....,0,0]编码
    print('无异常样本:',train_df[train_df['label']==''].shape[0])#2622
    #
    model_save_dir ='ckpt/'
    print_interval=-1
    train_batch_size=16
    val_batch_size=32
    max_epoch=21
    embed_num=859
    device = torch.device('cuda')
    criterion = torch.nn.BCEWithLogitsLoss()
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021).split(np.arange(train_df.shape[0]), train_df.label.values)
    kfold_best=[]
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # 
        print('train fold {}'.format(fold+1))
        model=CNN_Text(embed_num)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 ,weight_decay=5e-4)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
        train_dataset = textDataset(train_df,trn_idx)
        trainloader = DataLoader(train_dataset,
                                batch_size=train_batch_size,
                                shuffle=True,
                                num_workers=0)
        val_dataset = textDataset(train_df,val_idx)
        val_loader = DataLoader(val_dataset,
                                batch_size=val_batch_size,
                                shuffle=False,
                                num_workers=4)
        best_loss=train_model(model,criterion, optimizer,lr_scheduler=lr_scheduler)
        kfold_best.append(best_loss)
    print("local cv:",kfold_best,np.mean(kfold_best))