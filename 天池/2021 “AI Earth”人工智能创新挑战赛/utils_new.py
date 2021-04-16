import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import random
import netCDF4
#from eofs.standard import Eof

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

def mkdir(path):
    """
    mkdir of the path
    :param input: string of the path
    return: boolean
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path+' is created!')
        return True
    else:
        print(path+' already exists!')
        return False

def removeland(data):
    '''
    :param data: [N, 12, 24, 72]
    '''
    print('before nan number:', np.sum(np.isnan(data)))
    is_on_land = np.loadtxt('./lib/land_mask.txt')
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

def box(lonrange, latrange):
    lon_filt = (lon >= lonrange[0]) & (lon <= lonrange[1])
    lon_filt = np.arange(len(lon_filt))[lon_filt==True]
    lat_filt = (lat >= latrange[0]) & (lat <= latrange[1])
    lat_filt = np.arange(len(lat_filt))[lat_filt==True]
    return [min(lat_filt), max(lat_filt)+1], [min(lon_filt), max(lon_filt)+1]

masker = {
    'pacific': box([140, 280], [-20, 20]),
    'nino': box([190, 240], [-5, 5]),
    'wwv': box([120, 280], [-5, 5]),
    'wwv_left': box([120, 190], [-5, 5]),
    'wwv_right': box([240, 280], [-5, 5])
}

def get_area(data, name='nino'):
    '''
    :param data: [month, lon, lat]
    '''
    mask = masker[name]
    area = data[:, mask[0][0]:mask[0][1], mask[1][0]:mask[1][1], :]
    return area

class ninoDataset(Dataset):
    def __init__(self, path='../ENSO_COMP/enso_round1_train_20210201/', phase='train', p=0.8, extend=6, shuffle=True):

        # load data
        CMIP_data = netCDF4.Dataset(path+'CMIP_train.nc')
        SODA_data = netCDF4.Dataset(path+'SODA_train.nc')
        CMIP_label = netCDF4.Dataset(path+'CMIP_label.nc') 
        SODA_label = netCDF4.Dataset(path+'SODA_label.nc')

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
        print(self.X.shape)
        self.X = np.nan_to_num(self.X) # nan to zeros
        Y = np.concatenate(Y, axis=0)
        self.Y = (Y - maxmin['y'][1]) / (maxmin['y'][0] - maxmin['y'][1])
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
        self.define_idx(extend, I_train, I_test)
    
    def define_idx(self, extend, I_train, I_test):
        if self.phase == 'train':
            self.I = np.concatenate(I_train)
            self.max_sample=int(len(self.I)*extend) 
        else:
            self.I = np.concatenate(I_test)
            self.max_sample=int(len(self.I)*extend)
        
    def __getitem__(self, index):
        start_point=np.random.choice(self.I)

        x = self.X[start_point: start_point+12]
        y = self.Y[start_point+12: start_point+36]
        m = self.M[start_point: start_point+36]
        return {'X': {'x': torch.Tensor(x)[:, :, :, 0:1], 'm': torch.LongTensor(m)}, 'Y': torch.Tensor(y)}

    def __len__(self):
        return self.max_sample

class GraphDataset(ninoDataset):
    
    # def define_idx(self, extend, I_train, I_test):
    #     stride = [1, 4]
    #     if self.phase == 'train':
    #         self.I = np.concatenate(I_train)
    #         stride_I = []
    #         new_idx = np.cumsum([np.random.randint(stride[0], stride[1]) for _ in range(len(self.I))])
    #         new_idx = new_idx[new_idx <= len(self.I)]
    #         self.I = self.I[new_idx]
    #     else:

    def __getitem__(self, index):
        start_point=np.random.choice(self.I)

        x = self.X[start_point: start_point+12]
        y = self.Y[start_point+12: start_point+36]
        m = self.M[start_point: start_point+36]
        return {'X': {'x': torch.Tensor(x), 'm': torch.LongTensor(m)}, 'Y': torch.Tensor(y)}


class GraphEofDataset(ninoDataset):

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

    def __getitem__(self, index):
        start_point=np.random.choice(self.I)

        x = self.X[start_point: start_point+12]
        y = self.Y[start_point+12: start_point+36]
        m = self.M[start_point: start_point+36]
        e = np.concatenate([self.get_EOF(get_area(x[i:i+3, :, :, 0:1], 'pacific')[:, :, :, 0], 1, 'corr') for i in range(0, 12, 3)], axis=0)  # [4, lon, lat]
        e[np.isnan(e)] = 0.0
        return {'X': {'x': torch.Tensor(x), 'm': torch.LongTensor(m).cuda(), 'e': torch.Tensor(e).cuda()}, 'Y': torch.Tensor(y).cuda()}

class RollingDataset(ninoDataset):
    
    def __getitem__(self, index):
        start_point=np.random.choice(self.I)

        x = self.X[start_point: start_point+36]
        m = self.M[start_point: start_point+36]
        return {'x': torch.Tensor(get_area(x, 'nino')), 'm': torch.LongTensor(m)}

class ImageOutDataset(ninoDataset):
    
    def __getitem__(self, index):
        start_point=np.random.choice(self.I)

        x = self.X[start_point: start_point+12]
        m = self.M[start_point: start_point+36]
        y = self.X[start_point+12: start_point+36, :, :, 0:1]
        return {'x': torch.Tensor(get_area(x, 'nino')), 'm': torch.LongTensor(m), 'y': torch.squeeze(torch.Tensor(get_area(y, 'nino')), dim=3)}

if __name__ == '__main__':

    dataset = GraphEofDataset(path='../ENSO_COMP/enso_round1_train_20210201/', phase='train', p=0.8, extend=6, shuffle=True)
    print(len(dataset))
    trainloader = DataLoader(dataset, batch_size=16)
    for i, data in enumerate(trainloader):
        break