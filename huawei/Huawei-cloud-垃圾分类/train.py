#
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import Config
from models import resnet50
from utils import rubbishDataset
#
def train_model(model,criterion, optimizer):

    train_dataset = rubbishDataset(opt.train_val_data, opt.train_list, phase='train', input_size=opt.input_size)
    trainloader = DataLoader(train_dataset,
                             batch_size=opt.train_batch_size,
                             shuffle=True,
                             num_workers=opt.num_workers)

    total_iters=len(trainloader)
    model_name=opt.backbone
    train_loss = []
    since = time.time()
    best_score = 0.0
    best_epoch = 0
    #
    for epoch in range(1,opt.max_epoch+1):
        model.train(True)
        begin_time=time.time()
        running_corrects_linear = 0
        count=0
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

            if i % opt.print_interval == 0 or out_linear.size()[0] < opt.train_batch_size:
                spend_time = time.time() - begin_time
                print(
                    ' Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch, count, total_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * total_iters // 60 - spend_time // 60))
                train_loss.append(loss.item())
            running_corrects_linear += torch.sum(linear_preds == labels.data)
            #
        weight_score = val_model(model, criterion)
        epoch_acc_linear = running_corrects_linear.double() / total_iters / opt.train_batch_size
        print('Epoch:[{}/{}] train_acc={:.3f} '.format(epoch, opt.max_epoch,
                                                                    epoch_acc_linear))
        #
        model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + str(epoch) + '.pth'
        best_model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + 'best' + '.pth'
        #save the best model
        if weight_score > best_score:
            best_score = weight_score
            best_epoch=epoch
            torch.save(model.state_dict(), best_model_out_path)
            print("best epoch: {} best acc: {}".format(best_epoch,weight_score))
        #save based on epoch interval
        if epoch % opt.save_interval == 0 and epoch>opt.min_save_epoch:
            torch.save(model.state_dict(), model_out_path)
    #
    print('Best acc: {:.3f} Best epoch:{}'.format(best_score,best_epoch))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

@torch.no_grad()
def val_model(model, criterion):
    val_dataset = rubbishDataset(opt.train_val_data, opt.val_list, phase='val', input_size=opt.input_size)
    val_loader = DataLoader(val_dataset,
                             batch_size=opt.val_batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)
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
    #
    val_acc = accuracy_score(labels_list, pres_list)
    print('val_size: {}  valLoss: {:.4f} valAcc: {:.4f}'.format(dset_sizes, running_loss / dset_sizes,
                                                                      val_acc))
    return val_acc

if __name__ == "__main__":
    #
    opt = Config()
    torch.cuda.empty_cache()
    device = torch.device(opt.device)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    model_name=opt.backbone
    model_save_dir =os.path.join(opt.checkpoints_dir , model_name)
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, opt.num_classes)
    model.to(device)
    model = nn.DataParallel(model)
    optimizer = optim.SGD((model.parameters()), lr=opt.lr, momentum=opt.MOMENTUM, weight_decay=0.0004)
    train_model(model, criterion, optimizer)
