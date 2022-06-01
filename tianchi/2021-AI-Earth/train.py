import torch
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from nino_dataset import ninoDataset
from model import SpatailTimeNN
#from model_cnnLstm import SpatailTimeNN
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from sklearn.metrics import mean_squared_error
from pytorch_toolbelt import losses as L
from log import get_logger
import numpy as np
from metrics import rmse,score
from config import Config
def Upsample(x):
    Upsample_m=nn.UpsamplingNearest2d(scale_factor=2)#size=(120,360)
    x=Upsample_m(x)
    return x
def train_model(model,criterion, optimizer, lr_scheduler=None):

    total_iters=len(trainloader)
    logger.info('total_iters:{}'.format(total_iters))
    since = time.time()
    best_score = -1e10
    best_epoch = 0
    logger.info('start training...')
    #
    iters = len(trainloader)
    for epoch in range(1,opt.max_epoch+1):
        model.train(True)
        begin_time=time.time()
        logger.info('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        logger.info('Epoch {}/{}'.format(epoch,opt.max_epoch))
        logger.info('-' * 10)
        running_corrects_linear = 0
        count=0
        train_loss = []
        for i, data in enumerate(trainloader):
            count+=1
            inputs, labels,month = data['X']['x'],data['Y'],data['X']['m']
            x=inputs[0]
            labels = labels.type(torch.float).cuda()
            inputs=[x.cuda()]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新学习率
            if opt.scheduler=='cosine':
                lr_scheduler.step(epoch + count / iters)
            if i % opt.print_interval == 0 or outputs.size()[0] < opt.train_batch_size:
                spend_time = time.time() - begin_time
                logger.info(
                    ' Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
            #-
            train_loss.append(loss.item())
        if opt.scheduler!='cosine':
            lr_scheduler.step()
        val_score,val_rmse= val_model(model, criterion)
        logger.info('valrmse: {:.4f} valScore: {:.4f}'.format(val_rmse,val_score))
        #
        model_out_path = model_save_dir + "/" + '{}_'.format(opt.model_name) + str(epoch) + '.pth'
        best_model_out_path = model_save_dir + "/" + '{}_'.format(opt.model_name) + 'best' + '.pth'
        #save the best model
        if val_score > best_score:
            best_score = val_score
            best_epoch=epoch
            torch.save(model.state_dict(), best_model_out_path)
            logger.info("best epoch: {} best score: {}".format(best_epoch,val_score))
        #save based on epoch interval
        if epoch>opt.min_save_epoch and epoch%opt.save_epoch==0:
            torch.save(model.state_dict(), model_out_path)
    #
    logger.info('Best score: {:.3f} Best epoch:{}'.format(best_score,best_epoch))
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

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
        inputs, labels,month = data['X']['x'],data['Y'],data['X']['m']
        x=inputs[0]
        labels = labels.type(torch.float).cuda()
        inputs=[x.cuda()]
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        pres_list+=outputs.cpu().numpy().tolist()
        labels_list+=labels.data.cpu().numpy().tolist()
        running_loss += loss.item() * outputs.size(0)
        cont += 1
    #
    labels_arr=np.array(labels_list)
    pre_arr=np.array(pres_list)
    val_score = score(labels_arr, pre_arr)
    val_rmse = rmse(labels_arr, pre_arr)
    return val_score,val_rmse
if __name__=='__main__':
    #
    opt=Config()
    torch.cuda.empty_cache()
    device = torch.device('cuda')
    if opt.loss=='L1':
        criterion = torch.nn.L1Loss().cuda()
    if opt.loss=='MSE':
        criterion = torch.nn.MSELoss().cuda()
    model_save_dir =os.path.join(opt.checkpoints_dir , opt.model_name)
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    logger = get_logger(os.path.join(model_save_dir,'log.log'))
    logger.info('Using: {}'.format(opt.model_name))
    logger.info('use_frt: {} use_lstm_decoder:{}'.format(opt.use_frt,opt.use_lstm_decoder))
    logger.info('train_batch_size: {}'.format(opt.train_batch_size))
    logger.info('optimizer: {}'.format(opt.optimizer))
    logger.info('scheduler: {}'.format(opt.scheduler))
    logger.info('lr: {}'.format(opt.lr))
    logger.info('T_0:{} T_mult:{}'.format(opt.T_0,opt.T_mult))
    logger.info('p:{} extend:{}'.format(opt.p,opt.extend))
    model=SpatailTimeNN()
    model.to(device)
    #
    if opt.optimizer=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=opt.momentum, weight_decay=opt.weight_decay)
    if opt.scheduler=='cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.T_0, T_mult=opt.T_mult, eta_min=1e-6, last_epoch=-1)
    elif opt.scheduler=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=2)
    if opt.pretrain!=None:
        net_weight = torch.load(opt.pretrain)
        model.load_state_dict(net_weight)
        logger.info("load from :"+opt.pretrain)
        #scheduler=None
    #
    train_dataset = ninoDataset(
                    root=opt.root,
                    phase='train',
                    p=opt.p,
                    extend=opt.extend,
                    mode=opt.mode)
    trainloader = DataLoader(train_dataset,
                             batch_size=opt.train_batch_size,
                             shuffle=True,
                             num_workers=opt.num_workers)
    val_dataset = ninoDataset(
                    root=opt.root,
                    phase='test',
                    p=opt.p,
                    extend=opt.extend,
                    mode=opt.mode)
    val_loader = DataLoader(val_dataset,
                             batch_size=opt.val_batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)
    train_model(model, criterion, optimizer,lr_scheduler=scheduler)
    torch.cuda.empty_cache()