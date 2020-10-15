import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import pdb
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
import albumentations as A
from torch.nn import functional as F
from sklearn.metrics import f1_score,precision_recall_fscore_support, accuracy_score
from albumentations.pytorch import ToTensorV2
from albumentations import FancyPCA
from torch.utils.data import Dataset, DataLoader
from log import get_logger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import sunDataset, sunDatasetInfer
from cnn_finetune import make_model


torch.cuda.empty_cache()
GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)

TEST_DIR = 'test_input'
SUB_DIR='submitKfold/'
LOG_DIR='logKfold/'
if not os.path.exists(LOG_DIR):
   os.makedirs(LOG_DIR)
# 保存模型路径
OUT_DIR = 'output/'
#
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE= 8
TEST_BATCH_SIZE = 8
MOMENTUM = 0.9
NUM_EPOCHS = 12
LR = 0.001
VAL_INTERVAl = 1
# 打印间隔STEP
PRINT_INTERVAL = 150
# 最低保存模型/计算最优模型epohc阈值
MIN_SAVE_EPOCH = 5
def loaddata(train_dir, batch_size, shuffle,is_train=True):
    image_datasets = sunDataset(train_dir,is_train=is_train)
    dataset_loaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    data_set_sizes = len(image_datasets)
    return dataset_loaders, data_set_sizes

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS,model_name=None,train_dir=None,val_dir=None,fold=None):
    train_loss = []
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    model.train(True)
    logger.info('start training...')
    for epoch in range(1,NUM_EPOCHS+1):
        begin_time=time.time()
        data_loaders, dset_sizes = loaddata(train_dir=train_dir, batch_size=TRAIN_BATCH_SIZE, shuffle=True, is_train=True)
        logger.info('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        logger.info('-' * 10)
        optimizer = lr_scheduler(optimizer, epoch)
        running_loss = 0.0
        running_corrects = 0
        count=0
        for i, data in enumerate(data_loaders):
            count+=1
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % PRINT_INTERVAL == 0 or outputs.size()[0] < TRAIN_BATCH_SIZE:
                spend_time = time.time() - begin_time
                logger.info(' Epoch:{}({}/{}) loss:{:.3f} epoch_Time:{}min:'.format(epoch, count, dset_sizes // TRAIN_BATCH_SIZE,
                                                                         loss.item(),
                                                                         spend_time / count * dset_sizes / TRAIN_BATCH_SIZE // 60-spend_time//60))
        
                train_loss.append(loss.item())
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
           
        val_acc = test_model(model, criterion,val_dir)
        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes
        logger.info('Epoch:[{}/{}]\t Loss={:.5f}\t Acc={:.3f}'.format(epoch , NUM_EPOCHS, epoch_loss, epoch_acc))
        if val_acc > best_acc and epoch > MIN_SAVE_EPOCH:
            best_acc = val_acc
            best_model_wts = model.state_dict()
        if val_acc > 0.999:
            break
        save_dir = os.path.join(OUT_DIR,model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_out_path = save_dir + "/" + '{}_'.format(model_name+'Kfold'+str(fold))+str(epoch) + '.pth'
        if epoch % 1 == 0 and epoch > MIN_SAVE_EPOCH:
            torch.save(model, model_out_path)
    # save best model
    logger.info('Best Accuracy: {:.3f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + '{}_best.pth'.format(model_name+'Kfold'+str(fold))
    torch.save(model, model_out_path)
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model_out_path


def test_model(model, criterion,val_dir=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pres_list=[]
    labels_list=[]
    data_loaders, dset_sizes = loaddata(train_dir=val_dir, batch_size=VAL_BATCH_SIZE,  shuffle=False, is_train=False)
    for data in data_loaders:
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
    _,_, f_class, _= precision_recall_fscore_support(y_true=labels_list, y_pred=pres_list,labels=[0, 1, 2], average=None)                                                                   
    fper_class = {'beta': f_class[1], 'betax': f_class[2], 'alpha': f_class[0]}
    logger.info('各类单独F1:{}  各类F1取平均:{}'.format(fper_class, f_class.mean()))
    logger.info('val_size: {}  valLoss: {:.4f} valAcc: {:.4f}'.format(dset_sizes, running_loss / dset_sizes, running_corrects.double() / dset_sizes))
    return running_corrects.double() / dset_sizes
                                                  
def view_train_pic():
    data_loaders, _ = loaddata(train_dir=None, batch_size=TRAIN_BATCH_SIZE, shuffle=True, is_train=True)
    pic, label = next(iter(data_loaders))
    img = torchvision.utils.make_grid(pic)
    img = img.numpy().transpose([1,2,0])
    plt.imshow(img)
    plt.show()

def exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=5):
    LR = init_lr * (0.8**(epoch / lr_decay_epoch))
    logger.info('Learning Rate is {:.5f}'.format(LR))
    for param_group in optimizer.param_groups:
        param_group['LR'] = LR
    return optimizer

def infer(net_weight,model_name,fold=None):
    logger.info('start inference....')
    logger.info(10*'-')
    image_datasets =sunDatasetInfer(TEST_DIR)
    dataset_loaders = torch.utils.data.DataLoader(image_datasets,batch_size=TEST_BATCH_SIZE,shuffle=False, num_workers=0)
    # 加载最佳模型
    
    model= torch.load(net_weight)
    model.eval()
    pres_list=[]
    path_list=[]
    for data in dataset_loaders:
        inputs,paths = data
        paths=[name.split('.')[3].split('_')[1] for name in paths]
        inputs= Variable(inputs.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        pres_list+=preds.cpu().numpy().tolist()
        path_list+=paths
    pres_write=[str(p+1) for p in pres_list]
    if not os.path.exists(SUB_DIR):
        os.makedirs(SUB_DIR)
    w=open(os.path.join(SUB_DIR,'submit_{}.txt'.format(model_name+'Kfold'+str(fold))),'w')
    for k in range(1172):
        w.write(path_list[k]+' '+pres_write[k])
        if k!=1171:
            w.write('\n')
    w.close()
    #alpha,beta,betax
    re_dic={'alpha':0,'beta':0,'betax':0}
    for i in pres_list:
        if i==0:
            re_dic['alpha']+=1
        elif i==1:
            re_dic['beta']+=1
        else:
            re_dic['betax']+=1
    #这里将预测结果与线上90的结果作一个对比
    res90= {'alpha':0,'beta':0,'betax':0}
    pres90=[]
    with open('submit_merge90.txt') as f:
        for line in f.readlines():
            line=line.strip('\n')
            tmp=line.strip(' ')[-1]
            pres90.append(int(tmp)-1)
            if tmp=='1':
                res90['alpha']+=1
            elif tmp=='2':
                res90['beta']+=1
            else:
                res90['betax']+=1
    logger.info('{}: predict: {}'.format(model_name,re_dic))
    logger.info('submit_merge.txt: predict: {}'.format(res90))
    _, _, f_class, _ = precision_recall_fscore_support(y_true=pres90, y_pred=pres_list,average=None)
    score = accuracy_score(y_true=pres90, y_pred=pres_list)
    logger.info("{} 和线上90的 f1:{} Acc:{}".format(model_name, f_class, score))


if __name__ == "__main__":
    folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    model_name = 'se_resnet50'  # 'resnet18','xception','mobilenet_v2', 'shufflenet_v2_x1_0']:
    logger = get_logger(LOG_DIR + model_name+'.log')
    for fold in range(5):
        train_dir = 'dataset/' + folds[fold] + '/train'
        val_dir = 'dataset/' + folds[fold] + '/val'
        fold+=1
        logger.info('Using: {},Fold:{}'.format(model_name,fold))

        model  = make_model('{}'.format(model_name), num_classes=3, pretrained=True, input_size=(224,224))
        criterion = nn.CrossEntropyLoss().cuda()
        model = model.cuda()
        optimizer = optim.SGD((model.parameters()), lr=LR, momentum=MOMENTUM, weight_decay=0.0004)
        cos_scheduler = CosineAnnealingWarmRestarts(optimizer, 3)
        model_out_path= train_model(model, criterion, optimizer, lr_scheduler=exp_lr_scheduler
                                    , num_epochs=NUM_EPOCHS,model_name=model_name
                                    ,train_dir=train_dir,val_dir=val_dir,fold=fold)
        model_out_path=os.path.join('output/',model_name) + "/" + '{}_best.pth'.format(model_name+'Kfold'+str(fold))
        infer(model_out_path,model_name,fold)
        torch.cuda.empty_cache()


