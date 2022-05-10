import torch
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import pandas as pd
import torch.optim as optim
import pdb
#超参数
EPOCH=500
TIME_STEP=1#
INPUT_SIZE=int(28/TIME_STEP)#
LR=0.01
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
            return torch.from_numpy(sample_x).reshape(TIME_STEP,-1),torch.from_numpy(sample_y)
        else:
            return torch.from_numpy(sample_x).reshape(TIME_STEP,-1)
    def __len__(self):
        return len(self.data)

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn=torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )

        self.out=torch.nn.Linear(32,1)
    def forward(self, x):

        r_out,(h_n,h_c)=self.rnn(x,None)#r_out:[batch,5,64],r_out[:,-1,:]:[batch,64]
        out=self.out(r_out[:,-1,:])#[batch,1]
        return out

#
train_batch=32
val_batch=32
train_loader = DataLoader(GPSDataset('dataset/x_train.csv','dataset/y_train.csv'), batch_size=train_batch,
                            shuffle=True, num_workers=0)
test_loader = DataLoader(GPSDataset('dataset/x_test.csv','dataset/y_test.csv'), batch_size=val_batch,
                            shuffle=True, num_workers=0)
rnn=RNN()
print(rnn)
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,50,100,200,500,750], gamma=0.05)   #学习率按区间更新
loss_func=torch.nn.MSELoss()

for epoch in range(1,EPOCH+1):
    for step,(x,y) in enumerate(train_loader):
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        b_x=Variable(x)#(batch,time_step.input_size)
        b_y=Variable(y).squeeze()
        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        test_loss=0
        if step % 50==0:
            for test_x,test_y in test_loader:
                test_x, test_y = torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)
                test_output=rnn(x)
                #print(test_output.shape,test_y.shape,test_x.shape)
                test_loss += torch.nn.MSELoss(reduction='sum')(test_output,test_y).item()
            test_loss /= len(test_loader.dataset)
            print('Train Epoch: {} Train_loss: {:.4f},Val_loss: {:.4f}'.format(epoch,loss.item(),test_loss))