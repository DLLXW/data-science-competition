import timm
import os
import torch
import torch.nn as nn
import numpy as np
import zipfile
from utils_new import masker
from eofs.standard import Eof

def get_area(data, name='nino'):
    '''
    :param data: [month, lon, lat]
    '''
    mask = masker[name]
    area = data[:,:, mask[0][0]:mask[0][1], mask[1][0]:mask[1][1]]
    return area
def removeland(data):
    '''
    :param data: [N, 12, 24, 72]
    '''
    print('before nan number:', np.sum(np.isnan(data)))
    is_on_land = np.loadtxt('land_mask.txt')
    mask = np.zeros(data.shape, dtype=int)
    mask[:,:,:,:] = is_on_land[np.newaxis,np.newaxis,:,:]
    data[mask==1] = 0
    return data
def get_months(start):
    '''
    :param start: 1 to 12
    '''
    res = []
    tmp = start - 1
    for i in range(36):
        res.append(tmp)
        tmp += 1
        if tmp == 12:
            tmp = 0
    return res
#
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
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3,padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3,padding=1),
                nn.MaxPool2d(kernel_size=2),
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

#
def get_EOF(data, order=1, mode='corr'):
    '''
    :param data: image data, sst or t300, [month, lon, lat]
    :param order: int
    return: eof_corr, eof_cova [order, lon, lat],
            pc [month, order]
    '''
    solver = Eof(data)
    if mode == 'corr':
        res = solver.eofsAsCorrelation(neofs=order)
    elif mode == 'cova':
        res = solver.eofsAsCovariance(neofs=order)
    elif mode == 'pc':
        res = solver.pcs(npcs=order, pcscaling=1) 
    return res
def get_ftr_eof(sst):
    eof=[]
    for i in range(9):
        tmp=sst[i:i+3,:,:]
        re=get_EOF(tmp)
        re[np.isnan(re)] = 0
        eof.append(re)
    #
    eof=np.concatenate(eof)
    return eof
def get_frt(sst):#手工特征
    diff=[]
    for i in range(11):
        tmp=sst[:,i+1:i+2,:,:]-sst[:,i:i+1,:,:]
        diff.append(tmp)
    diff=torch.cat(diff,dim=1)
    win_mean=[]
    for i in range(9):
        tmp=sst[:,i:i+3,:,:]
        tmp=torch.mean(tmp,dim=1,keepdim=True)
        win_mean.append(tmp)
    win_mean=torch.cat(win_mean,dim=1)
    return diff,win_mean#[11,9,29],[4,9,29]
def Upsample(x):
    Upsample_m=nn.UpsamplingNearest2d(scale_factor=2)#size=(120,360)
    x=Upsample_m(x)
    return x
#打包目录为zip文件（未压缩）
def make_zip(source_dir='./result/', output_filename = 'result.zip'):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    source_dirs = os.walk(source_dir)
    print(source_dirs)
    for parent, dirnames, filenames in source_dirs:
        print(parent, dirnames)
        for filename in filenames:
            if '.npy' not in filename:
                continue
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径
            zipf.write(pathfile, arcname)
    zipf.close()
#
weight_path='./SpatailTimeNN_best.pth'
test_path = './tcdata/enso_round1_test_20210201/'
use_frt=[True,True,True,True]
save_dir='./result/'
device = torch.device('cuda')
if not os.path.exists(save_dir):os.makedirs(save_dir)
#------------
model=SpatailTimeNN()
model.to(device)
model.load_state_dict(torch.load(weight_path))
model.eval()
print('download model {} finished...............'.format(weight_path))
print('download testdata {} finished...............'.format(test_path))
### 1. 测试数据读取
files = os.listdir(test_path)
test_feas_dict = {}
for file_name in files:
    test_feas_dict[file_name] = np.load(test_path + file_name)
    
### 2. 结果预测
test_predicts_dict = {}
for file_name,val in test_feas_dict.items():
    val = np.expand_dims(val, axis=0)
    val=val[:,:,:,:,0]
    val=torch.from_numpy(removeland(val)).type(torch.FloatTensor)
    #val=get_area(val,name='pacific').type(torch.FloatTensor)
    print(val.shape)
    #
    with torch.no_grad():
        out_pre=model([val.cuda()])
    test_predicts_dict[file_name] = out_pre.cpu().numpy()[0]
#
### 3.存储预测结果
for file_name,val in test_predicts_dict.items(): 
    np.save('./result/' + file_name,val)
#
make_zip()