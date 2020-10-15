import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import time
import os
from sklearn.metrics import accuracy_score
from utils.log import get_logger
from cnn_finetune import make_model
from config import Config
from utils import retriDataset
from PIL import ImageFile
from modeling import LandmarkNet
from torch.optim.lr_scheduler import StepLR
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_model(model,criterion, optimizer, scheduler):

    train_dataset = retriDataset(opt.data_root, opt.train_list, phase='train', input_size=opt.input_size)
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
    best_acc = 0.0
    model.train(True)
    logger.info('start training...')
    #
    for epoch in range(1,opt.max_epoch+1):
        begin_time=time.time()
        logger.info('Epoch {}/{}'.format(epoch, opt.max_epoch))
        logger.info('-' * 10)
        optimizer = scheduler(optimizer, epoch)
        running_loss = 0.0
        running_corrects = 0
        count=0
        for i, data in enumerate(trainloader):
            count+=1
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()
            out= model(inputs,labels)
            loss=criterion(out, labels)
            _,preds=torch.max(out.data, 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % opt.print_interval == 0 or out.size()[0] < opt.train_batch_size:
                spend_time = time.time() - begin_time
                logger.info(' Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(epoch, count, total_iters,
                                                                         loss.item(),optimizer.param_groups[-1]['lr'],
                                                                         spend_time / count * total_iters // 60-spend_time//60))
                train_loss.append(loss.item())
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
           
        #begin validating
        val_acc = val_model(model,criterion)
        epoch_loss = running_loss / total_iters
        epoch_acc=running_corrects.double()/total_iters/opt.train_batch_size
        logger.info('Epoch:[{}/{}]\t Loss={:.5f}\t  Acc={:.3f}'.format(epoch , opt.max_epoch, epoch_loss,epoch_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
        save_dir = os.path.join(opt.checkpoints_dir,model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_out_path = save_dir + "/" + '{}_'.format(model_name)+str(epoch) + '.pth'
        if epoch % opt.save_interval == 0:
            torch.save(model.state_dict(), model_out_path)
    # save best model
    logger.info('Best Accuracy: {:.3f}'.format(best_acc))
    model_out_path = save_dir + "/" + '{}_best.pth'.format(model_name)
    torch.save(best_model_wts, model_out_path)
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


#  
def val_model(model,criterion):
    val_dataset = retriDataset(opt.data_root, opt.val_list, phase='val', input_size=opt.input_size)
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
        outputs = model(inputs,labels)
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
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        cont += 1
    valAcc0 = running_corrects.double() / len(val_dataset)
    valAcc1= accuracy_score(labels_list,pres_list)
    print(valAcc0,valAcc1)
    logger.info('val_size: {}  valLoss: {:.4f} valAcc: {:.4f}'.format(dset_sizes,running_loss/len(val_loader), valAcc1))
    return valAcc1
#

def exp_lr_scheduler(optimizer, epoch):
    lr = opt.lr * (0.8**(epoch / opt.lr_decay_epoch))
    logger.info('Learning Rate is {:.5f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == "__main__":
    #
    opt = Config()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    if opt.criterion == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    #
    model_name =opt.backbone
    #
    log_dir=os.path.join(opt.checkpoints_dir,model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir , model_name+'.log'))
    logger.info('Using: {}'.format(model_name))
    logger.info('Using Loss: {}'.format(opt.loss_module))
    logger.info('Using criterion: {}'.format(opt.criterion))
    logger.info('input_size: {}'.format(str(opt.input_size))) 
    logger.info('fc_dim: {}'.format(str(opt.fc_dim)))
    logger.info('lr: {}'.format(str(opt.lr)))
    logger.info('optimizer: {}'.format(str(opt.optimizer)))
    logger.info('Using the GPU: {}'.format(str(opt.gpu_id)))

    #
    model  = LandmarkNet(n_classes=opt.num_classes,
                 model_name=model_name,
                 pooling=opt.pooling,
                 loss_module=opt.loss_module,
                fc_dim=opt.fc_dim)
    model.to(device)
    model = nn.DataParallel(model)
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD((model.parameters()), lr=opt.lr, momentum=opt.momentum, weight_decay=0.0004)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    #
    scheduler=exp_lr_scheduler
    train_model(model,criterion, optimizer,
                        scheduler=scheduler,
                        )
    torch.cuda.empty_cache()


