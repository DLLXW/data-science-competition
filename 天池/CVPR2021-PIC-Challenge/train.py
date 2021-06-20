import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import time
import pandas as pd
#
import timm
from dataset import faceDataset
from config import Config
from networks import landmarkRegressNet,landmarkPfld
from PIL import ImageFile
from torch.cuda.amp import autocast, GradScaler
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging

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
def train_model(model,criterion, optimizer, scheduler):
    total_iters=len(trainloader)
    logger.info('total_iters:{}'.format(total_iters))
    model_name=opt.backbone
    train_loss = []
    since = time.time()
    best_model_wts = model.state_dict()
    best_loss = 1e6
    logger.info('start training...')
    #
    for epoch in range(1,opt.max_epoch+1):
        model.train(True)
        begin_time=time.time()
        logger.info('Epoch {}/{}'.format(epoch, opt.max_epoch))
        logger.info('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        count=0
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            count+=1
            inputs, labels = data['image'],data['label']
            image_left,image_mid,image_right=inputs
            image_left=image_left.cuda()
            image_mid=image_mid.cuda()
            image_right=image_right.cuda()
            labels = labels.cuda()
            inputs=[image_left,image_mid,image_right]
            # out= model(inputs,labels)
            # loss=criterion(out, labels)
            # _,preds=torch.max(out.data, 1)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            with autocast():
                output=model(image_left,image_mid,image_right)
                loss=criterion(output,labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #
            scheduler.step(epoch + count / total_iters)
            if i % opt.print_interval == 0 or output.size()[0] < opt.batch_size:
                spend_time = time.time() - begin_time
                logger.info(' Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(epoch, count, total_iters,
                                                                         loss.item(),optimizer.param_groups[-1]['lr'],
                                                                         spend_time / count * total_iters // 60-spend_time//60))
                train_loss.append(loss.item())
            running_loss += loss.item()
        scheduler.step()
        #begin validating
        val_loss = val_model(model,criterion)
        epoch_loss = running_loss / total_iters
        logger.info('Epoch:[{}/{}]\t Loss={:.5f}\t'.format(epoch , opt.max_epoch, epoch_loss))
        save_dir = os.path.join(opt.checkpoints_dir,model_name)
        model_out_best_path=save_dir + "/" + '{}_'.format(model_name)+'best' + '.pth'
        model_out_path = save_dir + "/" + '{}_'.format(model_name)+str(epoch) + '.pth'
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, model_out_best_path)
        #
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #
        if epoch % opt.save_interval == 0:
            torch.save(model.state_dict(), model_out_path)
        
    # save best model
    logger.info('Best Loss: {:.3f}'.format(best_loss))
    model_out_path = save_dir + "/" + '{}_best.pth'.format(model_name)
    torch.save(best_model_wts, model_out_path)
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# 
@torch.no_grad()
def val_model(model,criterion):
    dset_sizes=len(valloader)
    model.eval()
    running_loss = 0.0
    for data in valloader:
        inputs, labels = data['image'],data['label']
        image_left,image_mid,image_right=inputs
        image_left=image_left.cuda()
        image_mid=image_mid.cuda()
        image_right=image_right.cuda()
        labels = labels.cuda()
        
        outputs = model(image_left,image_mid,image_right)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    #
    logger.info('val_size: {}  valLoss: {:.4f}'.format(dset_sizes,running_loss/len(valloader)))
    return running_loss/len(valloader)
#


if __name__ == "__main__":
    #
    opt = Config()
    scaler=GradScaler()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    criterion = torch.nn.L1Loss()
    #
    trainset = faceDataset( root=opt.root,
                            list_dir=opt.train_list_dir,
                            input_size=opt.input_size,
                            transform = None
                            )

    trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size = opt.batch_size,
                num_workers = opt.num_workers,
                #pin_memory = True,
                shuffle = True,
                #drop_last = True
        )
    #
    valset = faceDataset(root=opt.root,
                        list_dir=opt.val_list_dir,
                        input_size=opt.input_size,
                        transform = None
                        )

    valloader = torch.utils.data.DataLoader(
                valset,
                batch_size = opt.batch_size,
                num_workers = opt.num_workers,
                shuffle = False
        )
    #
    model_name =opt.backbone
    log_dir=os.path.join(opt.checkpoints_dir,model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir , model_name+'.log'))
    logger.info('Using: {}'.format(model_name))
    logger.info('input_size: {}'.format(str(opt.input_size)))
    logger.info('lr: {}'.format(str(opt.scheduler_params['lr_start'])))
    logger.info('optimizer: {}'.format(str(opt.optimizer)))
    #关键点回归网络
    if model_name=='pfld':
        model =landmarkPfld()
    else:
        model  = landmarkRegressNet(
                 model_arch=opt.backbone,
                 landmarks=opt.landmarks,
                 pretrained=True)
    #
    model.to(device)
    #model = nn.DataParallel(model)
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD((model.parameters()), lr=opt.lr, momentum=opt.momentum, weight_decay=0.0004)
    else:
        optimizer =  torch.optim.AdamW(model.parameters(),
                                 lr = opt.scheduler_params['lr_start'])

    #
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=1e-6, last_epoch=-1)

    train_model(model,criterion, optimizer,
                        scheduler=scheduler,
                        )
    torch.cuda.empty_cache()