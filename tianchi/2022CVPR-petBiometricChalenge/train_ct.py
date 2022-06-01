from glob import glob
from itertools import groupby
import shutil
from sklearn.metrics import roc_auc_score
from numpy.random import seed
from sklearn.model_selection import GroupKFold, StratifiedKFold,KFold
import torch
from torch import nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datetime import datetime
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import torch.nn.functional as F
import datetime
from dataset import DogDataset
from model_ct import DogNet
from torch.cuda.amp import autocast, GradScaler
#from utils.ranger import Ranger
from torch.optim.lr_scheduler import _LRScheduler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn import metrics
#
DATA_ROOT = '../data/pet_biometric_challenge_2022/train/images/'
TRAIN_CSV = '../data/pet_biometric_challenge_2022/train/train_data.csv'
train_csv = pd.read_csv(TRAIN_CSV)
SIM_CSV = './similarity_100.csv'
df_sim = pd.read_csv(SIM_CSV)
# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label=1-label
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
def calc_roc_auc(pred_file, gt_file):
    #preds = pd.read_csv(pred_file)
    #gts = pd.read_csv(gt_file)
    preds=pred_file
    gts=gt_file

    scores = {}
    for i in range(len(preds)):
        if preds['imageA'][i] not in scores:
            scores[preds['imageA'][i]] = {}
        scores[preds['imageA'][i]][preds['imageB'][i]] = preds['prediction'][i]


    pred_scores, gt_scores = [], []
    for i in range(len(gts)):
        gt_scores.append(gts['label'][i])
        pred_scores.append(scores[gts['imageA'][i]][gts['imageB'][i]])

    pred_scores, gt_scores = np.asarray(pred_scores), np.asarray(gt_scores)

    auc_score = roc_auc_score(gt_scores, pred_scores)
    return auc_score
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
class Config(object):
    seed=42
    # swin_large_patch4_window7_224 'convnext_large_in22ft1k' tf_efficientnet_b5_ns,convnext_xlarge_in22ft1k
    backbone = 'convnext_small'
    num_classes = 2 #
    use_kfold=True #use_kfold=True 为线下五折交叉验证， use_kfold=False 为全量训练
    #
    loss = 'CrossEntropyLoss'#focal_loss/CrossEntropyLoss
    accum_iter=2
    verbose=1
    #
    in_channels=3
    #input_size = 224
    height_size = 224
    width_size = 224
    batch_size = 32  # batch size
    optimizer = 'adamW'#sam/adamW/ranger
    lr_scheduler='CosineLR'#CosineLR/StepLR/
    MOMENTUM = 0.9
    num_workers = 4  # how many workers for loading data
    warmup_epoch=3
    max_epoch = 33# T0=2:6/14/30/62 ; T0=3:9/21/45/93
    T0=2
    save_epoch={2:[5,13,29,61],3:[8,20,44,92]}
    lr_decay_epoch = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    val_interval = 1
    print_interval = 20
    save_interval = 1
    tensorboard_interval=50
    min_save_epoch=0
    load_from = None
    #
    checkpoints_dir = 'ckpt_0512_100_loss/'
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
def get_train_transforms(height_size,width_size):
    return A.Compose([
        A.Resize(height_size,width_size),#[height,width]
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])

def get_valid_transforms(height_size,width_size):
    return A.Compose([
        A.Resize(height_size,width_size),#[height,width]
        A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])
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
def prepare_dataloader(df, trn_idx=None, val_idx=None, data_root=None,use_kfold=False):
    collator = torch.utils.data.dataloader.default_collate
    if use_kfold==False:
        val_idx=[ii for ii in range(len(df)) if ii%2==0]
        trn_idx=[ii for ii in range(len(df)) if ii%2!=0]
        train_=df
        #train_=df.loc[trn_idx,:].reset_index(drop=True) 
        valid_=df.loc[val_idx,:].reset_index(drop=True) 
        valid_=df
        train_ds = DogDataset(train_,df_sim, data_root, transform=get_train_transforms(opt.height_size,opt.width_size),mode='train')
        valid_ds = DogDataset(valid_,df_sim, data_root, transform=get_valid_transforms(opt.height_size,opt.width_size),mode='val')
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=opt.batch_size,
            pin_memory=False,
            drop_last=False,
            shuffle=True,        
            num_workers=4,
            collate_fn=collator,
            #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
        )
        val_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=opt.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=False,
        )
    else:
        train_ = df.loc[trn_idx,:].reset_index(drop=True)
        valid_ = df.loc[val_idx,:].reset_index(drop=True) 
        #train_.to_csv("train_fold3_x.csv",index=False)
        #valid_.to_csv("valid_fold3_x.csv",index=False)
        train_ds = DogDataset(train_,df_sim, data_root, transform=get_train_transforms(opt.height_size,opt.width_size),mode='train')
        valid_ds = DogDataset(valid_,df_sim, data_root, transform=get_valid_transforms(opt.height_size,opt.width_size),mode='val')
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=opt.batch_size,
            pin_memory=False,
            drop_last=False,
            shuffle=True,        
            num_workers=4,
            collate_fn=collator,
            #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
        )
        val_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=opt.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=False,
        )
    return train_loader, val_loader
def train_one_epoch(fold,epoch, model, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()
    begin_time = time.time()
    running_loss = None
    epoch_iters = len(train_loader)
    #valid_one_epoch(0, 0, model, val_loader, device)
    for step, data in enumerate(train_loader):
        imgA=data['image_A'].to(device).float()
        imgB=data['image_B'].to(device).float()
        image_labels = data['label'].to(device).long()
        #
        with autocast():
            frt_A,frt_B = model(imgA,imgB)
            #loss_ce = criterion_1(outputs, image_labels)
            loss = criterion(frt_A,frt_B,image_labels)
            #loss=loss_ct+loss_ce
            scaler.scale(loss).backward()
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01
            if ((step + 1) %  opt.accum_iter == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                if scheduler is not None and schd_batch_update:
                    scheduler.step(epoch + step / epoch_iters)
                    if epoch>=opt.warmup_epoch:
                       scheduler.step(epoch-opt.warmup_epoch + step / epoch_iters)
                    else:
                       warmup_scheduler.step()
            if step % opt.print_interval == 0 or frt_A.size()[0] < opt.batch_size:
                spend_time = time.time() - begin_time
                logger.info(
                    'Fold:{} Epoch:[{}/{}]({}/{})loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        fold,epoch,opt.max_epoch, step, epoch_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * epoch_iters // 60 - spend_time // 60))
    if scheduler is not None and not schd_batch_update:
        scheduler.step()
def simplify(param):
    new_pre={}
    for k, v in param.items():
        name = k[7:]
        new_pre[name] = v
    return new_pre
# 
@torch.no_grad()
def valid_one_epoch(fold, epoch, model, val_loader, device, scheduler=None, schd_loss_update=False):
    global max_auc,min_loss
    model.eval()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    #pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    name_A_lst=[]
    name_B_lst=[]
    for step, data in enumerate(val_loader):
        imgA=data['image_A'].to(device).float()
        imgB=data['image_B'].to(device).float()
        nameA=data['name_A']
        nameB=data['name_B']
        name_A_lst+=nameA
        name_B_lst+=nameB
        #
        image_labels = data['label'].to(device).long()
        frt_A,frt_B = model(imgA,imgB)
        euclidean_distance = F.pairwise_distance(frt_A, frt_B)
        pres=1/euclidean_distance
        pres=pres.detach().cpu().numpy()
        #
        image_preds_all +=pres.tolist()
        image_targets_all += image_labels.detach().cpu().numpy().tolist()
        sample_num += image_labels.shape[0]  
    #------------------
    df_pre=pd.DataFrame(columns=['imageA','imageB','prediction'])
    df_gt=pd.DataFrame(columns=['imageA','imageB','label'])
    df_pre['imageA']=name_A_lst
    df_pre['imageB']=name_B_lst
    df_pre['prediction']=image_preds_all
    df_gt['imageA']=name_A_lst
    df_gt['imageB']=name_B_lst
    df_gt['label']=image_targets_all
    val_auc=calc_roc_auc(df_pre,df_gt)
    #
    image_preds_all=np.array(image_preds_all)
    image_targets_all=np.array(image_targets_all)
    # print(image_preds_all)
    # print(image_preds_all.shape,image_targets_all.shape)
    #val_auc=metrics.roc_auc_score(image_targets_all,image_preds_all)
    logger.info('valid auc = {:.4f}'.format(val_auc))
    if val_auc > max_auc:
        max_auc = val_auc
        logger.info('best val_auc: {} best_epoch: {} '.format(max_auc,epoch))
        if fold>-1:
            torch.save(simplify(model.state_dict()),'{}/{}_fold_{}_best.pth'.format(save_dir, model_name, fold))
        else:
            torch.save(simplify(model.state_dict()),'{}/{}_all_data_best.pth'.format(save_dir, model_name))
           
    
    if fold<0 and epoch>opt.min_save_epoch:
        torch.save(simplify(model.state_dict()),'{}/{}_all_data_epoch_{}.pth'.format(save_dir, model_name, epoch))
    
if __name__ == '__main__':
    #
    opt = Config()
    seed_everything(opt.seed)
    model_name=opt.backbone
    save_dir =os.path.join(opt.checkpoints_dir , model_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir,'log.log'))
    logger.info('Using: {}'.format(model_name))
    logger.info('height_size: {},opt.width_size: {}'.format(opt.height_size,opt.width_size))
    logger.info('batch size: {}'.format(opt.batch_size))
    logger.info('warmup_epoch: {}'.format(opt.warmup_epoch))
    logger.info('optimizer: {}'.format(opt.optimizer))
    logger.info('lr_scheduler: {}'.format(opt.lr_scheduler))
    logger.info('Using kfold: {}'.format(str(opt.use_kfold)))
    auc_lst=[]
    shutil.copy('./train.py',save_dir+'/train.py')
    #folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt.seed).split(np.arange(train_csv.shape[0]), train_csv.class_id.values)
    folds = GroupKFold(n_splits=5).split(
                    X=np.arange(train_csv.shape[0]),
                    y=None,#train_csv['dog ID'].values,
                    groups=train_csv['dog ID'].values)
    #folds=KFold(n_splits=5, shuffle=True, random_state=opt.seed).split(np.arange(train_csv.shape[0]))
    if opt.use_kfold:
        for fold, (trn_idx, val_idx) in enumerate(folds):
            #
            max_auc = 0.
            min_loss = 1e10
            print('Training with Fold {} started'.format(fold))
            print(len(trn_idx), len(val_idx))
            train_loader, val_loader = prepare_dataloader(train_csv, trn_idx, val_idx, data_root=DATA_ROOT,use_kfold=True)
            device = torch.device('cuda')
            model = DogNet(model_name, pretrained=True).to(device)
            #model.load_state_dict(torch.load(load_from.format(fold)))
            model= torch.nn.DataParallel(model)
            scaler = GradScaler()
            if opt.optimizer == 'adamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            elif opt.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            elif opt.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
            elif opt.optimizer == 'ranger':
                optimizer=Ranger(model.parameters(),lr=1e-3)
            if opt.lr_scheduler=='StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=4)
            elif opt.lr_scheduler=='CosineLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.T0, T_mult=2, eta_min=1e-6, last_epoch=-1)
                #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.max_epoch-opt.warmup_epoch+1, T_mult=1, eta_min=1e-6, last_epoch=-1)
            iter_per_epoch=len(train_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * opt.warmup_epoch,start_lr=1e-6)
            #分类
            #criterion_1 = nn.CrossEntropyLoss()
            criterion=ContrastiveLoss()
            warmup_scheduler.step(0)
            for epoch in range(opt.max_epoch):
                train_one_epoch(fold,epoch, model, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=True)
                with torch.no_grad():
                   valid_one_epoch(fold, epoch, model, val_loader, device, scheduler=None, schd_loss_update=False)
                torch.save(simplify(model.state_dict()),'{}/{}_fold_{}_last.pth'.format(save_dir, model_name, fold))
                
            del model, optimizer, train_loader, val_loader, scaler, scheduler
            torch.cuda.empty_cache()
            auc_lst.append(max_auc)

        if len(auc_lst) > 1:
            nfold = len(auc_lst)
            logger.info({'k fold' : auc_lst})
            logger.info({str(nfold) + 'fold' : np.mean(auc_lst)})
    else:
        max_auc = 0.
        min_loss = 1e10
        print('---------------------Training with  all data --------------------- ')
        train_loader, val_loader = prepare_dataloader(train_csv, trn_idx=None, val_idx=None, data_root=DATA_ROOT)
        device = torch.device('cuda')
        model = ImgClassifier(model_name, 248, pretrained=True).to(device)
        model_path='ckpt/tf_efficientnet_b5_ns_pretrain/tf_efficientnet_b5_ns_pretrained43.pth'
        #model=load_model(model, model_path)
        model= torch.nn.DataParallel(model)
        #model.load_state_dict(torch.load("./2021-09-11-14-08/swin_base_patch4_window12_384_fold_0_best"))
        scaler = GradScaler()
        if opt.optimizer == 'adamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        elif opt.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        elif opt.optimizer == 'ranger':
            optimizer=Ranger(model.parameters(),lr=1e-3)
        if opt.lr_scheduler=='StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=4)
        elif opt.lr_scheduler=='CosineLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6, last_epoch=-1)
        
        criterion=nn.MultiLabelSoftMarginLoss()
        for epoch in range(10):
            train_one_epoch(-1,epoch, model, criterion,optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=True)
            with torch.no_grad():
                valid_one_epoch(-1, epoch, model, criterion, val_loader, device, scheduler=None, schd_loss_update=False)
            #
            torch.save(simplify(model.state_dict()),'{}/{}_all_data_last.pth'.format(save_dir, model_name))
            
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()