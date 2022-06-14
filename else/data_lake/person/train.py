from glob import glob
import shutil

from numpy.random import seed
from sklearn.model_selection import GroupKFold, StratifiedKFold,KFold
import torch
from torch import nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from datetime import datetime
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from utils.dataset_albu import RubbishDataset,get_train_transforms,get_valid_transforms
from net import ImgClassifier
from torch.cuda.amp import autocast, GradScaler
##from cutmix import CutMixCollator
from utils.ranger import Ranger

from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import _LRScheduler
import argparse
train_csv = pd.read_csv('./train_df_fusai.csv')
DATA_ROOT='../data/train2_new/'
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
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
class Config(object):
    seed=42#convnext_xlarge_in22ft1k
    backbone = 'convnext_large_in22ft1k'# swin_large_patch4_window7_224 'convnext_large_in22ft1k' tf_efficientnet_b5_ns,swin_base_patch4_window12_384,vit_base_patch16_384
    num_classes = 46 #
    use_kfold=True #use_kfold=True 为线下五折交叉验证， use_kfold=False 为全量训练
    #
    mixup=True
    alpha=0.2
    cutmix=False
    #
    smooth_ratio=0.2
    t1=1.
    t2=1.
    smooth_ratio=0.2
    #
    loss = 'CrossEntropyLoss'#focal_loss/CrossEntropyLoss
    accum_iter=2
    verbose=1
    #
    in_channels=3
    #input_size = 224
    height_size = 224
    width_size = 224
    batch_size = 16  # batch size
    optimizer = 'adamW'#sam/adamW/ranger
    lr_scheduler='CosineLR'#CosineLR/StepLR/
    MOMENTUM = 0.9
    num_workers = 0  # how many workers for loading data
    warmup_epoch = 2
    max_epoch = 10# T0=2:6/14/30/62 ; T0=3:9/21/45/93
    T0=3
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
    checkpoints_dir = 'ckpt_reproduct/'

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
    #from catalyst.data.sampler import BalanceClassSampler
    # if opt.do_cutmix:
    #     collator = CutMixCollator(opt.alpha_cutmix)
    # else:
    #     collator = torch.utils.data.dataloader.default_collate
    collator = torch.utils.data.dataloader.default_collate
    if use_kfold==False:
        val_idx=[ii for ii in range(len(df)) if ii%2==0]
        trn_idx=[ii for ii in range(len(df)) if ii%2!=0]
        train_=df
        #train_=df.loc[trn_idx,:].reset_index(drop=True) 
        valid_=df.loc[val_idx,:].reset_index(drop=True) 
        valid_=df
        train_ds = RubbishDataset(train_, data_root, transforms=get_train_transforms(opt.height_size,opt.width_size), output_label=True)
        valid_ds = RubbishDataset(valid_, data_root, transforms=get_valid_transforms(opt.height_size,opt.width_size), output_label=True)
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
        train_ds = RubbishDataset(train_, data_root, transforms=get_train_transforms(opt.height_size,opt.width_size), output_label=True)
        valid_ds = RubbishDataset(valid_, data_root, transforms=get_valid_transforms(opt.height_size,opt.width_size), output_label=True)
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
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    w=x.size()[-1]
    #label_len=y.shape[-1]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    #(n c h w)
    shuffle_x=x[index, :]
    #shuffle_y=y[index, :]
    #lam=random.choice([0.25,0.5])
    lam=0.5
    #print(x[:,:,:,:w//2].shape,y[:,:label_len//2].shape)
    mixed_x = torch.cat((x[:,:,:,:int(w*lam)],shuffle_x[:,:,:,int(w*lam):]),dim=3)
    #mixed_y = torch.cat((y[:,:label_len//lam],shuffle_y[:,label_len//lam:]),dim=1)
    #print(mixed_x.shape,mixed_y.shape)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)  
def train_one_epoch(fold,epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()
    begin_time = time.time()
    running_loss = None
    epoch_iters = len(train_loader)
    for step, (imgs, image_labels) in enumerate(train_loader):
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        #
        with autocast():
            # image_preds = model(imgs)
            # #分类Loss
            # loss = loss_fn(image_preds, image_labels)
            #-------mixup-------
            # if opt.mixup:
            #     imgs, targets_a, targets_b, lam = mixup_data(imgs, image_labels,
            #                                         opt.alpha)
            #     outputs = model(imgs)
            #     loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)
            if opt.cutmix:
                    if np.random.randn()<0.5:
                        #print('using mixup ')
                        imgs, targets_a, targets_b, lam = mixup_data(imgs, image_labels,
                                                        opt.alpha)
                        outputs = model(imgs)
                        loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)
                    else:
                        imgs, targets_a, targets_b, lam = cutmix_data(imgs, image_labels)
                        outputs = model(imgs)
                        loss = cutmix_criterion(loss_fn, outputs, targets_a, targets_b, lam)
                        #print('using cutmix........ ')
            elif opt.mixup:
                imgs, targets_a, targets_b, lam = mixup_data(imgs, image_labels,
                                                    opt.alpha)
                outputs = model(imgs)
                loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(imgs)
                loss = loss_fn(outputs, image_labels)
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
                    if epoch>=opt.warmup_epoch:
                        scheduler.step(epoch-opt.warmup_epoch + step / epoch_iters)
                    else:
                        warmup_scheduler.step()
            if step % opt.print_interval == 0 or outputs.size()[0] < opt.batch_size:
                spend_time = time.time() - begin_time
                logger.info(
                    'Fold:{} Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
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
# 计算准确率
def calculat_f1(output, target):
    '''
    {'upperLength': {'LongSleeve': 0, 'ShortSleeve': 1, 'NoSleeve': 2}, 
    'clothesStyles': {'Solidcolor': 0, 'multicolour': 1, 'lattice': 2}, 
    'hairStyles': {'Short': 0, 'Long': 1, 'middle': 2, 'Bald': 3}, 
    'lowerLength': {'Skirt': 0, 'Trousers': 1, 'Shorts': 2}, 
    'lowerStyles': {'Solidcolor': 0, 'lattice': 1, 'multicolour': 2}, 
    'shoesStyles': {'Sandals': 0, 'Sneaker': 1, 'LeatherShoes': 2, 'else': 3}, 
    'towards': {'right': 0, 'back': 1, 'front': 2, 'left': 3}}
    '''
    upperLength_output=output[:,:3]
    clothesStyles_output=output[:,3:6]
    hairStyles_output=output[:,6:10]
    lowerLength_output=output[:,10:13]
    lowerStyles_output=output[:,13:16]
    shoesStyles_output=output[:,16:20]
    towards_output=output[:,20:24]
    color1_output=output[:,24:35]
    color2_output=output[:,35:46]
    #
    upperLength_target=target[:,:3]
    clothesStyles_target=target[:,3:6]
    hairStyles_target=target[:,6:10]
    lowerLength_target=target[:,10:13]
    lowerStyles_target=target[:,13:16]
    shoesStyles_target=target[:,16:20]
    towards_target=target[:,20:24]
    color1_target=target[:,24:35]
    color2_target=target[:,35:46]
    #
    upperLength_output = nn.functional.softmax(upperLength_output, dim=1)
    upperLength_output = torch.argmax(upperLength_output, dim=1)
    upperLength_target = torch.argmax(upperLength_target, dim=1)
    upperLength_f1 = f1_score(upperLength_target, upperLength_output, average='macro')
    #
    clothesStyles_output = nn.functional.softmax(clothesStyles_output, dim=1)
    clothesStyles_output = torch.argmax(clothesStyles_output, dim=1)
    clothesStyles_target = torch.argmax(clothesStyles_target, dim=1)
    clothesStyles_f1 = f1_score(clothesStyles_target, clothesStyles_output, average='macro')
    #
    hairStyles_output = nn.functional.softmax(hairStyles_output, dim=1)
    hairStyles_output = torch.argmax(hairStyles_output, dim=1)
    hairStyles_target = torch.argmax(hairStyles_target, dim=1)
    hairStyles_f1 = f1_score(hairStyles_target, hairStyles_output, average='macro')
    #
    lowerLength_output = nn.functional.softmax(lowerLength_output, dim=1)
    lowerLength_output = torch.argmax(lowerLength_output, dim=1)
    lowerLength_target = torch.argmax(lowerLength_target, dim=1)
    lowerLength_f1 = f1_score(lowerLength_target, lowerLength_output, average='macro')
    #
    lowerStyles_output = nn.functional.softmax(lowerStyles_output, dim=1)
    lowerStyles_output = torch.argmax(lowerStyles_output, dim=1)
    lowerStyles_target = torch.argmax(lowerStyles_target, dim=1)
    lowerStyles_f1 = f1_score(lowerStyles_target, lowerStyles_output, average='macro')
    #
    shoesStyles_output = nn.functional.softmax(shoesStyles_output, dim=1)
    shoesStyles_output = torch.argmax(shoesStyles_output, dim=1)
    shoesStyles_target = torch.argmax(shoesStyles_target, dim=1)
    shoesStyles_f1 = f1_score(shoesStyles_target, shoesStyles_output, average='macro')
    #
    towards_output = nn.functional.softmax(towards_output, dim=1)
    towards_output = torch.argmax(towards_output, dim=1)
    towards_target = torch.argmax(towards_target, dim=1)
    towards_f1 = f1_score(towards_target, towards_output, average='macro')
    #
    color1_output = nn.functional.softmax(color1_output, dim=1)
    color1_output = torch.argmax(color1_output, dim=1)
    color1_target = torch.argmax(color1_target, dim=1)
    color1_f1 = f1_score(color1_target, color1_output, average='macro')
    #
    color2_output = nn.functional.softmax(color2_output, dim=1)
    color2_output = torch.argmax(color2_output, dim=1)
    color2_target = torch.argmax(color2_target, dim=1)
    color2_f1 = f1_score(color2_target, color2_output, average='macro')
    #
    f1_list = [upperLength_f1,clothesStyles_f1,hairStyles_f1,lowerLength_f1,lowerStyles_f1,shoesStyles_f1,towards_f1,color1_f1,color2_f1]
    return f1_list
@torch.no_grad()
def valid_one_epoch(fold, epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    global max_f1,min_loss
    model.eval()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    #pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in enumerate(val_loader):
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        image_preds = model(imgs)
        image_preds_all +=image_preds.detach().cpu().numpy().tolist()
        image_targets_all += image_labels.detach().cpu().numpy().tolist()
        # loss = loss_fn(image_preds, image_labels)
        # loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]  
    image_preds_all=torch.from_numpy(np.array(image_preds_all))
    image_targets_all=torch.from_numpy(np.array(image_targets_all))
    f1_list=calculat_f1(image_preds_all,image_targets_all)
    val_f1=np.mean(f1_list)
    logger.info('valid f1 = {:.4f} f1_list = {}'.format(val_f1,f1_list))
    if val_f1 > max_f1:
        max_f1 = val_f1
        logger.info('best val_f1: {} best_epoch: {} '.format(max_f1,epoch))
        if fold>-1:
            torch.save(simplify(model.state_dict()),'{}/{}_fold_{}_best.pth'.format(save_dir, model_name, fold))
        else:
            torch.save(simplify(model.state_dict()),'{}/{}_all_data_best.pth'.format(save_dir, model_name))
           
    
    if fold<0 and epoch>opt.min_save_epoch:
        torch.save(simplify(model.state_dict()),'{}/{}_all_data_epoch_{}.pth'.format(save_dir, model_name, epoch))
#
if __name__ == '__main__':
    #
    parser=argparse.ArgumentParser()
    parser.add_argument('--backbone',type=str,default='swin_large_patch4_window7_224',help='')
    args=parser.parse_args()
    model_name=args.backbone
    opt = Config()
    seed_everything(opt.seed)
    save_dir =os.path.join(opt.checkpoints_dir , model_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir,'log.log'))
    logger.info('Using: {}'.format(model_name))
    logger.info('height_size: {},opt.width_size: {}'.format(opt.height_size,opt.width_size))
    logger.info('batch size: {}'.format(opt.batch_size))
    logger.info('criterion: {}'.format(opt.loss))
    logger.info('smooth_ratio: {}'.format(opt.smooth_ratio))
    logger.info('optimizer: {}'.format(opt.optimizer))
    logger.info('lr_scheduler: {}'.format(opt.lr_scheduler))
    logger.info('Using kfold: {}'.format(str(opt.use_kfold)))
    logger.info('Using mixup: {} alpha: {}'.format(opt.mixup,opt.alpha))
    f1_lst=[]
    shutil.copy('./train.py',save_dir+'/train.py')
    #folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt.seed).split(np.arange(train_csv.shape[0]), train_csv.class_id.values)
    # folds = GroupKFold(n_splits=5).split(np.arange(train_csv.shape[0]),
    #                 train_csv.class_id.values,
    #                 train_csv.group.values)
    folds=KFold(n_splits=100, shuffle=True, random_state=opt.seed).split(np.arange(train_csv.shape[0]))
    if opt.use_kfold:
        for fold, (trn_idx, val_idx) in enumerate(folds):
            #
            if fold>0:
                break
            max_f1 = 0.
            min_loss = 1e10
            print('Training with Fold {} started'.format(fold))
            print(len(trn_idx), len(val_idx))
            train_loader, val_loader = prepare_dataloader(train_csv, trn_idx, val_idx, data_root=DATA_ROOT,use_kfold=True)
            device = torch.device('cuda')
            model = ImgClassifier(model_name,opt.num_classes, pretrained=True).to(device)
            #model.load_state_dict(torch.load(load_from.format(fold)))
            model= torch.nn.DataParallel(model)
            scaler = GradScaler()
            if opt.optimizer == 'adamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            elif opt.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            elif opt.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
            elif opt.optimizer == 'ranger':
                optimizer=Ranger(model.parameters(),lr=1e-3)
            if opt.lr_scheduler=='StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=4)
            elif opt.lr_scheduler=='CosineLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.max_epoch-opt.warmup_epoch+1, T_mult=1, eta_min=1e-6, last_epoch=-1)
            #多标签分类
            iter_per_epoch=len(train_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * opt.warmup_epoch,start_lr=1e-6)
            criterion = nn.MultiLabelSoftMarginLoss()
            for epoch in range(opt.max_epoch):
                train_one_epoch(fold,epoch, model, criterion, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=True)
                with torch.no_grad():
                   valid_one_epoch(fold, epoch, model, criterion, val_loader, device, scheduler=None, schd_loss_update=False)
                torch.save(simplify(model.state_dict()),'{}/{}_fold_{}_last.pth'.format(save_dir, model_name, fold))
                
            del model, optimizer, train_loader, val_loader, scaler, scheduler
            torch.cuda.empty_cache()
            f1_lst.append(max_f1)

        if len(f1_lst) > 1:
            nfold = len(f1_lst)
            logger.info({'k fold' : f1_lst})
            logger.info({str(nfold) + 'fold' : np.mean(f1_lst)})
    else:
        max_f1 = 0.
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        
        criterion=nn.MultiLabelSoftMarginLoss()
        for epoch in range(10):
            train_one_epoch(-1,epoch, model, criterion,optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=True)
            with torch.no_grad():
                valid_one_epoch(-1, epoch, model, criterion, val_loader, device, scheduler=None, schd_loss_update=False)
            #
            torch.save(simplify(model.state_dict()),'{}/{}_all_data_last.pth'.format(save_dir, model_name))
            
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()