import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision import transforms as T
import torchvision
import netCDF4
class ninoDataset(Dataset):

    def __init__(self,root='',mode='CMIP',phase='train'):
        self.root=root
        self.mode=mode
        self.phase = phase
        self.tr_fea,self.val_fea,self.tr_label,self.val_label=self.get_frt()
        #
        if self.phase == 'train':
            self.data=self.tr_fea
            self.label=self.tr_label
        else:
            self.data=self.val_fea
            self.label=self.val_label
    def get_frt(self):
        label_path=self.root+'/'+self.mode+"_label.nc"
        data_path=self.root+'/'+self.mode+"_train.nc"
        nc_label= netCDF4.Dataset(label_path,'r')
        tr_nc_labels=nc_label['nino'][:]
        #
        nc_data=netCDF4.Dataset(data_path,'r') 
        nc_sst=np.array(nc_data['sst'][:])
        nc_t300=np.array(nc_data['t300'][:])
        nc_ua=np.array(nc_data['ua'][:])
        nc_va=np.array(nc_data['va'][:])
        #
        # print(nc_sst.shape)
        # print(nc_t300.shape)
        # print(tr_nc_labels.shape)
        ### 训练特征，保证和训练集一致
        tr_features = np.concatenate([nc_sst[:,:12,:,:].reshape(-1,12,24,72,1),nc_t300[:,:12,:,:].reshape(-1,12,24,72,1),\
                                    nc_ua[:,:12,:,:].reshape(-1,12,24,72,1),nc_va[:,:12,:,:].reshape(-1,12,24,72,1)],axis=-1)
        #
        tr_features[np.isnan(tr_features)] = -0.0376
        #print(np.nanmax(tr_features),np.nanmin(tr_features))
        ### 训练标签，取后24个
        tr_labels = tr_nc_labels[:,12:] 

        ### 训练集验证集划分
        tr_len     = int(tr_features.shape[0] * 0.8)
        tr_fea     = tr_features[:tr_len,:].copy()
        tr_label   = tr_labels[:tr_len,:].copy()
        val_len     = tr_features.shape[0]-tr_len
        val_fea     = tr_features[tr_len:,:].copy()
        val_label   = tr_labels[tr_len:,:].copy()
        #
        tr_fea=torch.from_numpy(tr_fea)
        val_fea=torch.from_numpy(val_fea)
        tr_label=torch.from_numpy(tr_label)
        val_label=torch.from_numpy(val_label)
        tr_fea=tr_fea.permute(0,1,4,2,3).reshape(tr_len,-1,24,72)
        val_fea=val_fea.permute(0,1,4,2,3).reshape(val_len,-1,24,72)
        return tr_fea,val_fea,tr_label,val_label
        #

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data.float(), label

    def __len__(self):
        return len(self.data)
if __name__ == '__main__':
    #
    dataset = ninoDataset(
                    root='../enso_round1_train_20210201',
                    mode='CMIP',
                    phase='test')

    trainloader = DataLoader(dataset, batch_size=2)
    for i, (data, label) in enumerate(trainloader):
        #
        Upsample_m=nn.UpsamplingNearest2d(scale_factor=5)#size=(120,360)
        data=Upsample_m(data)
        #
        print(data.shape,label.shape)
        break