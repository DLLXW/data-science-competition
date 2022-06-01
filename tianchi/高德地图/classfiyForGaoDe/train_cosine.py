import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import time
import os
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
from utils.log import get_logger
from cnn_finetune import make_model
from config import Config
from models import *
from utils import gaodeDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchcontrib.optim import SWA
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_model(model,criterion, optimizer, lr_scheduler,arc_model=None):

    train_dataset = gaodeDataset(opt.trainValConcat_dir, opt.train_list, phase='train', input_size=opt.input_size)
    trainloader = DataLoader(train_dataset,
                             batch_size=opt.train_batch_size,
                             shuffle=True,
                             num_workers=opt.num_workers)

    total_iters=len(trainloader)
    logger.info('total_iters:{}'.format(total_iters))
    model_name=opt.backbone
    train_loss = []
    since = time.time()
    best_model_wts = model.state_dict()
    best_score = 0.0
    model.train(True)
    logger.info('start training...')
    #
    if lr_scheduler is cos_lr_scheduler:
        return_lr_scheduler = lr_scheduler(optimizer, 4)
    if lr_scheduler is exp_lr_scheduler:
        return_lr_scheduler = lr_scheduler(optimizer)
    #
    optimizer=SWA(optimizer,swa_start=10, swa_freq=5)
    for epoch in range(1,opt.max_epoch+1):
        begin_time=time.time()
        #logger.info('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
        logger.info('Epoch {}/{}'.format(epoch, opt.max_epoch))
        logger.info('-' * 10)
        #optimizer = lr_scheduler(optimizer, epoch)
        running_loss = 0.0
        running_corrects_linear = 0
        running_corrects_arc=0
        count=0
        iters=len(trainloader)
        for i, data in enumerate(trainloader):
            count+=1
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()
            #
            out_linear= model(inputs)
            _, linear_preds = torch.max(out_linear.data, 1)
            loss = criterion(out_linear, labels)
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is cos_lr_scheduler:
                return_lr_scheduler.step(epoch + i / iters)

            if i % opt.print_interval == 0 or out_linear.size()[0] < opt.train_batch_size:
                spend_time = time.time() - begin_time
                logger.info(
                    ' Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch, count, total_iters,
                        loss.item(),optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
                train_loss.append(loss.item())
            running_corrects_linear += torch.sum(linear_preds == labels.data)
            #
        if lr_scheduler is exp_lr_scheduler:
            return_lr_scheduler.step()
        weight_score = val_model(model, criterion)
        epoch_acc_linear = running_corrects_linear.double() / total_iters / opt.train_batch_size
        logger.info('Epoch:[{}/{}] train_acc={:.3f} '.format(epoch, opt.max_epoch,
                                                                    epoch_acc_linear))
        save_dir = os.path.join(opt.checkpoints_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_out_path = save_dir + "/" + '{}_'.format(model_name) + str(epoch) + '.pth'
        best_model_out_path = save_dir + "/" + '{}_'.format(model_name) + 'best' + '.pth'
        #save the best model
        if weight_score > best_score:
            best_score = weight_score
            torch.save(model.state_dict(), best_model_out_path)
        #save based on epoch interval
        if epoch % opt.save_interval == 0 and epoch>opt.min_save_epoch:
            torch.save(model.state_dict(), model_out_path)
        #
        optimizer.swap_swa_sgd
    #
    logger.info('Best WeightF1: {:.3f}'.format(best_score))
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def val_model(model, criterion):
    val_dataset = gaodeDataset(opt.trainValConcat_dir, opt.val_list, phase='val', input_size=opt.input_size)
    val_loader = DataLoader(val_dataset,
                             batch_size=opt.val_batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)
    dset_sizes=len(val_loader)
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
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        pres_list+=preds.cpu().numpy().tolist()
        labels_list+=labels.data.cpu().numpy().tolist()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1
    _, _, f_class, _ = precision_recall_fscore_support(y_true=labels_list, y_pred=pres_list, labels=[0, 1, 2, 3],
                                                       average=None)
    fper_class = {'畅通': f_class[0], '缓行': f_class[1], '拥堵': f_class[2],'封闭':f_class[3]}
    weight_score=0.1 * f_class[0] + 0.2 * f_class[1] + 0.3 * f_class[2] + 0.4 * f_class[3]
    val_acc = accuracy_score(labels_list, pres_list)
    logger.info('各类单独F1:{}  各类F加权:{}'.format(fper_class, weight_score))
    logger.info('val_size: {}  valLoss: {:.4f} valAcc: {:.4f}'.format(dset_sizes, running_loss / dset_sizes,val_acc))
    return weight_score

'''
def exp_lr_scheduler(optimizer, epoch):
    LR = opt.LR * (0.8**(epoch / opt.lr_decay_epoch))
    logger.info('Learning Rate is {:.5f}'.format(LR))
    for param_group in optimizer.param_groups:
        param_group['LR'] = LR
    return optimizer

'''


def cos_lr_scheduler(optimizer,t_max=3):
    return CosineAnnealingLR(optimizer, T_max=t_max)

def exp_lr_scheduler(optimizer, step_size=5):
    return StepLR(optimizer, step_size, gamma=0.9)
if __name__ == "__main__":
    #
    opt = Config()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    #
    model_name =opt.backbone
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    logger = get_logger(os.path.join(opt.log_dir , model_name+'.log'))
    logger.info('Using: {}'.format(model_name))
    logger.info('InputSize: {}'.format(opt.input_size))
    logger.info('optimizer: {}'.format(opt.optimizer))
    logger.info('lr_init: {}'.format(opt.LR))
    logger.info('criterion: {}'.format(opt.loss))
    logger.info('lr_scheduler: {}'.format(opt.lr_scheduler))
    logger.info('Using the GPU: {}'.format(str(opt.gpu_id)))

    model  = make_model('{}'.format(model_name), num_classes=opt.num_classes,
                        pretrained=True, input_size=(opt.input_size,opt.input_size))
    model.to(device)
    model = nn.DataParallel(model)
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD((model.parameters()), lr=opt.LR, momentum=opt.MOMENTUM, weight_decay=0.0004)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.LR)

    if opt.lr_scheduler=='cosine':
        lr_scheduler=cos_lr_scheduler
    else:
        lr_scheduler=exp_lr_scheduler
    train_model(model, criterion, optimizer,
              lr_scheduler=lr_scheduler)

    torch.cuda.empty_cache()
