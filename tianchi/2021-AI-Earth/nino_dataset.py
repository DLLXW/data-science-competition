import numpy as np
import os
from tqdm import tqdm
import torch
import netCDF4
from torch.utils.data import Dataset, DataLoader
import random
from config import  Config
from utils_new import get_area
from eofs.standard import Eof
maxmin = {
'x':[[10.198975563049316, -16.549121856689453], 
    [9.318188667297363, -11.912577629089355], 
    [15.002762794494629, -22.261333465576172], 
    [17.425098419189453, -17.876197814941406]],
'y':[4.138188362121582, -3.5832221508026123]
}

lon = np.array([  0.,   5.,  10.,  15.,  20.,  25.,  30.,  35.,  40.,  45.,  50.,
        55.,  60.,  65.,  70.,  75.,  80.,  85.,  90.,  95., 100., 105.,
       110., 115., 120., 125., 130., 135., 140., 145., 150., 155., 160.,
       165., 170., 175., 180., 185., 190., 195., 200., 205., 210., 215.,
       220., 225., 230., 235., 240., 245., 250., 255., 260., 265., 270.,
       275., 280., 285., 290., 295., 300., 305., 310., 315., 320., 325.,
       330., 335., 340., 345., 350., 355.], dtype=np.int)

lat = np.array([-55., -50., -45., -40., -35., -30., -25., -20., -15., -10.,  -5.,
         0.,   5.,  10.,  15.,  20.,  25.,  30.,  35.,  40.,  45.,  50.,
        55.,  60.], dtype=np.int)
def removeland(data):
    '''
    :param data: [N, 12, 24, 72]
    '''
    print('before nan number:', np.sum(np.isnan(data)))
    is_on_land = np.loadtxt('land_mask.txt')
    mask = np.zeros(data.shape, dtype=int)
    mask[:,:,:,:] = is_on_land[np.newaxis,np.newaxis,:,:]
    data[mask==1] = np.nan
    return data

def concat_all(data, label):
    data_all = [data[0]] + [data[i, 24: , :, :, :] for i in range(1, data.shape[0])]
    data_all = np.concatenate(data_all, axis=0)

    label_all = [label[0]] + [label[i, 24:] for i in range(1, label.shape[0])]
    label_all = np.concatenate(label_all, axis=0)

    month_all = [list(range(12)) for j in range(3)] + [list(range(12)) for i in range(1, label.shape[0])]  # 0 ~ 11
    month_all = np.concatenate(month_all, axis=0)
    return data_all, label_all, month_all
    
class ninoDataset(Dataset):
    def __init__(self, root='../enso_round1_train_20210201/', phase='train', p=0.8, extend=1, shuffle=False,mode='CNN'):
        
        self.mode=mode
        # load data
        CMIP_data = netCDF4.Dataset(root+'CMIP_train.nc')
        SODA_data = netCDF4.Dataset(root+'SODA_train.nc')
        CMIP_label = netCDF4.Dataset(root+'CMIP_label.nc') 
        SODA_label = netCDF4.Dataset(root+'SODA_label.nc')

        # select 4 channels
        feaname = ['sst', 't300', 'ua', 'va']
        CMIP_data = np.stack([removeland(np.array(CMIP_data.variables[i][:])) for i in feaname], axis=4)  # [year, 36, 24, 72, 4] (replace land into nan)
        SODA_data = np.stack([removeland(np.array(SODA_data.variables[i][:])) for i in feaname], axis=4)  # [year, 36, 24, 72, 4] 
        CMIP_label = np.array(CMIP_label.variables['nino']) # [year, 36]
        SODA_label = np.array(SODA_label.variables['nino']) # [year, 36]

        X_soda, Y_soda, M_soda = concat_all(SODA_data, SODA_label)
        X, Y, M, I = [X_soda], [Y_soda], [M_soda], [np.arange(len(X_soda))]
        # year 1-2265:      151 years *15 modes = 2265
        # year 2266-4645:   140 years *17 modes = 2380
        data1, label1 = CMIP_data[0: 2265], CMIP_label[0: 2265]
        for i in np.arange(0, 2265, 151):
            tmp_x, tmp_y, tmp_m = concat_all(data1[i: i+151], label1[i: i+151])
            X.append(tmp_x)
            Y.append(tmp_y)
            M.append(tmp_m)
            I.append(np.arange(len(tmp_x)) + I[-1][-1] + 1)  # add the new index

        data2, label2 = CMIP_data[2265:], CMIP_label[2265:]
        for i in np.arange(0, 2380, 140):
            tmp_x, tmp_y, tmp_m = concat_all(data2[i: i+140], label2[i: i+140])
            X.append(tmp_x)
            Y.append(tmp_y)
            M.append(tmp_m)
            I.append(np.arange(len(tmp_x)) + I[-1][-1] + 1)

        self.X = np.concatenate(X, axis=0)
        self.X = np.nan_to_num(self.X) # nan to zeros
        self.Y = np.concatenate(Y, axis=0)
        self.M = np.concatenate(M, axis=0)
         # prevent the overlap given the start idx
        I_train, I_test = [], []
        for idx in range(len(I)):
            I[idx] = I[idx][:-36+1]  # remove the last 35 index
            if shuffle:
                np.random.seed(idx)  # random seed
                np.random.shuffle(I[idx])  # shuffle
            I_train.append(I[idx][0: int(len(I[idx])*p)])
            I_test.append(I[idx][int(len(I[idx])*p): ])

        self.phase = phase
        total_len=self.X.shape[0]

        if self.phase == 'train':
            self.I = np.concatenate(I_train)
            self.max_sample=int(len(self.I)*extend) 
        else:
            self.I = np.concatenate(I_test)
            self.max_sample=int(len(self.I)*extend)
    def get_EOF(self, data, order=1, mode='corr'):
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
    def get_frt(self,sst):#手工特征
        diff=[]
        for i in range(11):
            tmp=sst[i+1:i+2,:,:]-sst[i:i+1,:,:]
            diff.append(tmp)
        diff=torch.cat(diff,dim=0)
        win_mean=[]
        for i in range(9):
            tmp=sst[i:i+3,:,:]
            tmp=torch.mean(tmp,dim=0,keepdim=True)
            win_mean.append(tmp)
        win_mean=torch.cat(win_mean,dim=0)
        return diff,win_mean#[11,9,29],[9,9,29]
    def get_ftr_eof(self,sst):
        eof=[]
        for i in range(9):
            tmp=sst[i:i+3,:,:]
            re=self.get_EOF(tmp)
            re[np.isnan(re)] = 0
            eof.append(re)
        #
        eof=np.concatenate(eof)
        return eof
    def __getitem__(self, index):
        start_point=np.random.choice(self.I)
        x = self.X[start_point: start_point+12]
        #x=get_area(x,name='pacific')
        #eof=self.get_ftr_eof(x[:,:,:,0])
        #eof=torch.from_numpy(eof).type(torch.FloatTensor)
        x=torch.from_numpy(x[:,:,:,0]).type(torch.FloatTensor)
        #diff,win_mean=self.get_frt(x)
        y = self.Y[start_point+12: start_point+36]
        m = self.M[start_point: start_point+36]
        return {'X': {'x':[x], 'm': torch.LongTensor(m)}, 'Y': torch.from_numpy(y)}

    def __len__(self):
        return self.max_sample



if __name__ == '__main__':

    dataset = ninoDataset(root='../enso_round1_train_20210201/', phase='train', mode='xxCNN')

    trainloader = DataLoader(dataset, batch_size=16)
    for i, data in enumerate(trainloader):
        #
        x=data['X']['x'][0]
        diff=data['X']['x'][1]
        eof=data['X']['x'][2]
        print(x.shape,diff.shape,eof.shape)
        print(eof)
        #print([[k,data['X'][k].shape] for k in list(data['X'].keys())])
        #rint(data['Y'].shape)
        break