import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import torch.optim as optim
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
torch.set_default_tensor_type(torch.FloatTensor)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28, 32),
            nn.BatchNorm1d(32),
            nn.ReLU6(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU6(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU6(),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, 28)
        x=self.net(x)
        x = self.fc(x)
        return x  #
class GPSDataset(Dataset):
    def __init__(self, x_path,label_path=None):
        self.x_path=x_path
        self.label_path = label_path
        self.data=pd.read_csv(self.x_path)
        if self.label_path!=None:
            self.label = pd.read_csv(self.label_path)
    def __getitem__(self,index):
        #第index个样本
        sample_x = self.data.iloc[index].values
        if self.label_path!=None:
            sample_y= self.label.iloc[index].values
            return torch.from_numpy(sample_x),torch.from_numpy(sample_y)
        else:
            return torch.from_numpy(sample_x)
    def __len__(self):
        return len(self.data)
def init_net(net):
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net
#
def train():
    lr=1e-4
    momentum=0.5
    epochs=1000
    device = torch.device("cpu")
    save_dir='output/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    eval_interval=50#验证间隔
    save_interval = 25  # 保存间隔
    writer=SummaryWriter()#用于记录训练和测试的loss
    #19903个训练样本，2111验证样本
    train_batch=128#全批次训练
    val_batch=512#全部验证
    train_loader = DataLoader(GPSDataset('dataset/x_train.csv','dataset/y_train.csv'), batch_size=train_batch,
                            shuffle=True, num_workers=0)
    test_loader = DataLoader(GPSDataset('dataset/x_test.csv','dataset/y_test.csv'), batch_size=val_batch,
                            shuffle=True, num_workers=0)
    model=MLP()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    milestones=[i*20 for i in range(5)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)   #学习率按区间更新
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_function=torch.nn.MSELoss()
    model.train()
    log_loss=0
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)  #
            loss.backward()

            optimizer.step()  # 根据parameter的梯度更新parameter的值
            # 下面是模型验证过程
            if batch_idx % eval_interval==0:
                model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():  # 无需计算梯度
                    for data, target in test_loader:
                        data, target = torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        test_loss += torch.nn.MSELoss(reduction='sum')(output, target).item()  # sum of batch loss
                test_loss /= len(test_loader.dataset)
                writer.add_scalars('loss', {'train_loss': loss, 'val_loss': test_loss}, global_step=log_loss)
                log_loss += 1
                print('Train Epoch: {} Train_loss: {:.4f},Val_loss: {:.4f}'.format(epoch,loss.item(),test_loss))
                model.train()
        scheduler.step()
        #保存模型
        if epoch%save_interval==0:
            torch.save(model.state_dict(), os.path.join(save_dir,str(epoch)+'.pth'))
    writer.close()

if __name__ == '__main__':
    train()

#