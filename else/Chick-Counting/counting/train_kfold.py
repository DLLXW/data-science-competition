
import argparse
import logging
import os
from pandas._config import config
import torch
from sklearn.model_selection import KFold, StratifiedKFold
import random
import numpy as np
import pandas as pd
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from utils.log import get_logger
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from  models.vgg import vgg19
from datasets.crowd_kfold import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from utils.ranger import  Ranger
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] ='0'  # set vis gpu
class Config(object):
    crop_size=512
    use_kfold=False
    downsample_ratio=8
    background_ratio=1.0
    sigma=8.0
    use_background=True
    batch_size = 8  # batch size
    optimizer = 'adamW'#sam/adamW/ranger
    lr_scheduler='CosineLR'#CosineLR/StepLR/MutiStepLR
    max_epoch = 30# T0=2:6/14/30/62 ; T0=3:9/21/45/93
    T0=3
    save_epoch={2:[5,13,29,61],3:[8,20,44,92]}
    lr_decay_epoch = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    val_interval = 1
    print_interval = 20
    save_interval = 1
    min_save_epoch=5
    load_from = None
    #
    checkpoints_dir = 'ckpt/'
#
SEED=512
train_csv = pd.read_csv('../data_counting/train_val_df_140.csv')
DATA_ROOT='../data_counting/140'
pretrained_path='./vgg19-dcbb9e9d.pth'
# DATA_ROOT_VAl='/home/trojanjet/baidu_qyl/tianma/data/ann_test'
# val_csv=pd.read_csv('../data/val_df_9.csv')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes
def prepare_dataloader(df, trn_idx=None, val_idx=None, data_root=None, data_root_val=None,use_kfold=True):
    if use_kfold:
        train_=df.loc[trn_idx,:].reset_index(drop=True) 
        valid_=df.loc[val_idx,:].reset_index(drop=True)
        #valid_=val_csv
        train_ds = Crowd(train_, data_root,opt.crop_size,opt.downsample_ratio,method='train')
        valid_ds = Crowd(valid_, data_root,opt.crop_size,opt.downsample_ratio,method='val')
    else:
        train_=train_csv
        #valid_=val_csv
        valid_=train_csv
        train_ds = Crowd(train_, data_root,opt.crop_size,opt.downsample_ratio,method='train')
        valid_ds = Crowd(valid_, data_root,opt.crop_size,opt.downsample_ratio,method='val')
    #
    train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=opt.batch_size,
            pin_memory=False,
            drop_last=False,
            shuffle=True,        
            num_workers=4,
            collate_fn=train_collate,
        )
    val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
        )
    return train_loader,val_loader
def train_eopch(fold,epoch,model,post_prob,criterion,dataloader,optimizer,scheduler=None, schd_batch_update=False):
    epoch_loss = AverageMeter()
    epoch_mae = AverageMeter()
    epoch_mse = AverageMeter()
    epoch_start = time.time()
    epoch_iters = len(dataloader)
    model.train()  # Set model to training mode
    # Iterate over data.
    for step, (inputs, points, targets, st_sizes) in enumerate(dataloader):
        inputs = inputs.to(device)
        st_sizes = st_sizes.to(device)
        gd_count = np.array([len(p) for p in points], dtype=np.float32)
        points = [p.to(device) for p in points]
        targets = [t.to(device) for t in targets]
        #with torch.set_grad_enabled(True):
        outputs = model(inputs)
        prob_list = post_prob(points, st_sizes)
        loss = criterion(prob_list, targets, outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        N = inputs.size(0)
        pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
        res = pre_count - gd_count
        epoch_loss.update(loss.item(), N)
        epoch_mse.update(np.mean(res * res), N)
        epoch_mae.update(np.mean(abs(res)), N)
        if scheduler is not None and schd_batch_update:
                    scheduler.step(epoch + step / epoch_iters)

    #logger.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
    #             .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
        #                    time.time()-epoch_start))
    if scheduler is not None and not schd_batch_update:
        scheduler.step()
    logger.info('lr: {}'.format(optimizer.param_groups[-1]['lr']))
@torch.no_grad()
def val_epoch(fold,epoch,model,dataloader):
    global best_acc,best_mae
    epoch_start = time.time()
    model.eval()  # Set model to evaluate mode
    epoch_res = []
    acc_lst=[]
    # Iterate over data.
    for inputs, count, _ in dataloader:
        inputs = inputs.to(device)
        # inputs are images with different sizes
        assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        outputs = model(inputs)
        target_cnt=count[0].item()
        res = count[0].item() - torch.sum(outputs).item()
        acc=1-abs(res)/(target_cnt+0.1)
        acc_lst.append(acc)
        epoch_res.append(res)
    #
    epoch_res = np.array(epoch_res)
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    acc=np.mean(acc_lst)
    #logger.info('Fold {} Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, ACC: {:.2f}, Cost {:.1f} sec'
     #               .format(fold,epoch, mse, mae, acc,time.time()-epoch_start))

    #
    logger.info('cur epoch acc: {} cur_epoch: {} '.format(acc,epoch))
    if acc>best_acc:
        best_acc=acc
        logger.info('best acc: {} best_epoch: {} '.format(best_acc,epoch))
        if fold>-1:
            torch.save(simplify(model.state_dict()),'{}/{}_fold_{}_best.pth'.format(save_dir, 'acc', fold))
        else:
            torch.save(simplify(model.state_dict()),'{}/{}_all_data_best.pth'.format(save_dir, 'acc'))
    if mae<best_mae:
        best_mae = mae
        # logger.info('best mae: {} best_epoch: {} '.format(best_mae,epoch))
        # if fold>-1:
        #     torch.save(simplify(model.state_dict()),'{}/{}_fold_{}_best.pth'.format(save_dir, 'mae', fold))
        # else:
        #     torch.save(simplify(model.state_dict()),'{}/{}_all_data_best.pth'.format(save_dir, 'mae'))
    


def simplify(param):
    new_pre={}
    for k, v in param.items():
        name = k[7:]
        new_pre[name] = v
    return new_pre

if __name__ == '__main__':
    opt = Config()
    opt.use_kfold=True
    seed_everything(SEED)
    logging=get_logger
    save_dir =os.path.join(opt.checkpoints_dir,'140_docker_train')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir,'log.log'))
    logger.info('use_kfold: {}'.format(opt.use_kfold))
    logger.info('crop_size: {}'.format(opt.crop_size))
    logger.info('batch size: {}'.format(opt.batch_size))
    logger.info('optimizer: {}'.format(opt.optimizer))
    logger.info('lr_scheduler: {}'.format(opt.lr_scheduler))
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED).split(np.arange(train_csv.shape[0]))
    acc_lst=[]
    if opt.use_kfold:
        for fold, (trn_idx, val_idx) in enumerate(folds):
            # we'll train fold 0 first
            best_acc = -1e3
            best_mae = 1e10
            print('Training with Fold {} started'.format(fold))
            logger.info("trn_idx:{} val_idx:{}".format(len(trn_idx),len(val_idx)))
            # if fold>2:
            #     cv=train_csv.loc[val_idx,:].reset_index(drop=True)
            #     cv.to_csv('{}.csv'.format(fold),index=False)
            train_loader, val_loader = prepare_dataloader(train_csv, trn_idx, val_idx, data_root=DATA_ROOT,data_root_val=DATA_ROOT,use_kfold=opt.use_kfold)
            device = torch.device('cuda')
            model =vgg19(pretrained_path=pretrained_path).to(device)
            model= torch.nn.DataParallel(model)
            if opt.optimizer == 'adamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=opt.weight_decay)
            elif opt.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
            elif opt.optimizer == 'ranger':
                optimizer=Ranger(model.parameters(),lr=5e-6)
            if opt.lr_scheduler=='StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=10)
            elif opt.lr_scheduler=='CosineLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.max_epoch+1, T_mult=1, eta_min=1e-6, last_epoch=-1)
            elif opt.lr_scheduler=='MultiStepLR':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35,45],gamma=0.2)
            #
            post_prob = Post_Prob(opt.sigma,
                                    opt.crop_size,
                                    opt.downsample_ratio,
                                    opt.background_ratio,
                                    opt.use_background,
                                    device)
            criterion = Bay_Loss(opt.use_background, device)
            for epoch in range(opt.max_epoch):
                train_eopch(fold,epoch,model,post_prob,criterion,train_loader,optimizer,scheduler=scheduler,schd_batch_update=True)

                with torch.no_grad():
                    val_epoch(fold,epoch,model,val_loader)
                #torch.save(simplify(model.state_dict()),'{}/{}_fold_{}_last.pth'.format(save_dir,'vgg', fold))
                
            del model, optimizer, train_loader, val_loader, scheduler
            torch.cuda.empty_cache()
            acc_lst.append(best_acc)
        logger.info('acc_lst {} mean acc {}'.format(acc_lst,np.mean(acc_lst)))
    else:
        best_acc = -1e3
        best_mae = 1e10
        print('---------------------Training with  all data --------------------- ')
        train_loader, val_loader = prepare_dataloader(train_csv, trn_idx=None, val_idx=None, data_root=DATA_ROOT,data_root_val=DATA_ROOT_VAl,use_kfold=opt.use_kfold)
        device = torch.device('cuda')
        model = vgg19(pretrained_path=pretrained_path).to(device)
        model= torch.nn.DataParallel(model)
        if opt.optimizer == 'adamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        elif opt.optimizer == 'ranger':
            optimizer=Ranger(model.parameters(),lr=1e-4)
        if opt.lr_scheduler=='StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=10)
        elif opt.lr_scheduler=='CosineLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.T0, T_mult=2, eta_min=1e-6, last_epoch=-1)
        elif opt.lr_scheduler=='MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,45],gamma=0.2)
        #
        post_prob = Post_Prob(opt.sigma,
                                opt.crop_size,
                                opt.downsample_ratio,
                                opt.background_ratio,
                                opt.use_background,
                                device)
        criterion = Bay_Loss(opt.use_background, device)
        for epoch in range(opt.max_epoch):
            train_eopch(-1,epoch,model,post_prob,criterion,train_loader,optimizer,scheduler=None,schd_batch_update=True)
            with torch.no_grad():
                val_epoch(-1,epoch,model,val_loader)
            #
            torch.save(simplify(model.state_dict()),'{}/{}_all_data_last.pth'.format(save_dir, 'vgg'))
            
        del model, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()