# -*- encoding: utf-8 -*-
'''
@File    :   cross_entropy_labelsmooth.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/8 15:03   xin      1.0         None
'''

import torch
from torch import nn
from collections import OrderedDict
import numpy as np


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss





class EdgeLabelSmoothing(nn.Module):
    def __init__(self,win_size=11,num_classes=5,smoothing=0.05):
        super(EdgeLabelSmoothing, self).__init__()
        assert (win_size%2) == 1
        self.smoothing = smoothing /(num_classes-1)
        self.win_size = win_size
        self.num_classes = num_classes
        self.find_edge_Conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=win_size,padding=(win_size-1)//2,stride=1,bias=False)
        
        new_state_dict = OrderedDict()
        weight = torch.zeros(1,1,win_size,win_size)
        weight = weight -1
        weight[:,:,win_size//2,win_size//2] = win_size*win_size - 1
        new_state_dict['weight'] = weight
        self.find_edge_Conv.load_state_dict(new_state_dict)
        self.find_edge_Conv.weight.requires_grad=False

    def to_categorical(self,y,alpha=0.05,num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical = categorical + alpha
        categorical[np.arange(n), y] = (1-alpha) + (alpha/self.num_classes)
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        categorical = categorical.transpose(0,3,1,2)
        return categorical
        

    def forward(self, x, target):
        assert x.size(1) == self.num_classes
        log_p = nn.functional.log_softmax(x,dim=1)
        target = target.unsqueeze(dim=1)
        self.find_edge_Conv.cuda(device=target.device)
        edge_mask = self.find_edge_Conv(target.float())
        edge_mask = edge_mask.data.cpu().numpy()
        edge_mask[edge_mask!=0] = 1
        self.smoothing = np.mean(edge_mask)
        if self.smoothing > 0.2:
            self.smoothing = 0.2
        
        target = target.squeeze(dim=1)
        target = target.data.cpu().numpy()
        onehot_mask = self.to_categorical(target,0,num_classes=self.num_classes)
        onehot_mask = onehot_mask*(1-edge_mask)
        softlabel_mask = self.to_categorical(target,alpha=self.smoothing,num_classes=self.num_classes)
        softlabel_mask = softlabel_mask*edge_mask
        onehot_mask = torch.from_numpy(onehot_mask).cuda(device=log_p.device).float()
        softlabel_mask = torch.from_numpy(softlabel_mask).cuda(device=log_p.device).float()
        loss = torch.sum(onehot_mask*log_p+softlabel_mask*log_p,dim=1).mean()
        return -loss