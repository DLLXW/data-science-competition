import timm
import torch
import torch.nn as nn
from config import Config
import numpy as np
from eofs.standard import Eof
class SpatailTimeNN(nn.Module):
    def __init__(self):
        super(SpatailTimeNN, self).__init__()
        self.conv_lst=nn.ModuleList([self.convPool(i) for i in range(12,0,-1)])
        self.predict_head_lst=nn.ModuleList([self.pre_head() for i in range(11)])
        #
        dim_cnn=12*27*11
        dim_encoder=128
        self.predict_head=nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(dim_cnn, 24),
                nn.Dropout(p=0.5),
                nn.Linear(24, 24)
                )
    def convPool(self,in_channels):#定义三层卷积提取特征
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=3,padding=1),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3,padding=1),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3,padding=1),
                nn.AvgPool2d(kernel_size=2),
            )
    def pre_head(self):
        return nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(12*27, 24),
                nn.Dropout(p=0.5),
                nn.Linear(24, 24)
                )
    def slice_win(self,x):
        x_lst=[]
        for i in range(11):
            x_lst.append(x[:,i:,:,:])#将输入滑窗，构造多个输入片段
        return x_lst
    def forward(self, inputs):
        cat_frt=[]#输入的特征
        x=inputs[0]
        x=self.slice_win(x)
        out=[]
        for i in range(len(x)):#len(x)
            x_win=x[i]
            x_win=self.conv_lst[i](x_win)#
            x_win=torch.flatten(x_win,start_dim=1)#
            cat_frt.append(x_win)
            out.append(self.predict_head_lst[i](x_win))
        #
        x=torch.cat(cat_frt,dim=-1)
        out_win=self.predict_head(x)
        out.append(out_win)
        out=torch.stack(out)
        out=torch.mean(out,axis=0)
        return out

if __name__=="__main__":
    opt=Config()
    device = torch.device('cuda')
    sst=torch.randn(2,12,24,72).cuda()
    # diff=torch.randn(2,11,9,29).cuda()
    # win_mean=torch.randn(2,9,9,29).cuda()
    eof=torch.randn(2,9,24,72).cuda()
    # sst=Upsample(sst)
    # diff=Upsample(diff)
    # win_mean=Upsample(win_mean)
    month=torch.LongTensor([[1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12],
            [1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12]]).cuda()
    area=torch.randn(2,12,90).cuda()
    
    model=SpatailTimeNN()
    print(model)
    model.to(device)
    out=model([sst,eof])
    print(out.shape)
    a=np.random.randn(1,2,3)