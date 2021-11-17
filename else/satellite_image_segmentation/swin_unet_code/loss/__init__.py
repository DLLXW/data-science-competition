# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:19   xin      1.0         None
'''

import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from PIL import Image


from .cross_entropy_labelsmooth import CrossEntropyLabelSmooth
from .lovasz_losses import lovasz_softmax, symmetic_lovasz
from .softce import SoftCrossEntropyLoss
from .dice import DiceLoss
from .large_margin_loss import LargeMarginSoftmaxV1


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.cpu().float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


def multi_scale_loss_fun(loss_fun, multi_scale_outputs, targets, device):
    if isinstance(multi_scale_outputs,tuple) or isinstance(multi_scale_outputs,list):
        _, _, H, W = multi_scale_outputs[0].shape
        labels_ = resize_labels(targets, size=(H, W))
        # labels_ = F.interpolate(targets.unsqueeze(1), (H, W), mode='nearest')
        loss_v = loss_fun(multi_scale_outputs[0], labels_.to(device))
        
        for i in range(1, len(multi_scale_outputs)):
            _, _, H, W = multi_scale_outputs[i].shape
            labels_ = resize_labels(targets, size=(H, W))
            loss_v = loss_v + loss_fun(multi_scale_outputs[i], labels_.to(device))
    else:
        _, _, H, W = multi_scale_outputs.shape
        labels_ = resize_labels(targets, size=(H, W))
        # labels_ = F.interpolate(targets.unsqueeze(1), (H, W), mode='nearest')
        loss_v = loss_fun(multi_scale_outputs, labels_.to(device))
    return loss_v


class Base_Loss(nn.modules.loss._Loss):
    def __init__(self, device, num_classes, loss_type='ce', class_weight=None, ignore_index=255):
        super(Base_Loss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.loss_type = loss_type
        if class_weight is not None:
            self.class_weight = torch.from_numpy(np.array(class_weight)).float()
        else:
            self.class_weight = class_weight
        self.ignore_index = ignore_index

    def forward(self, outputs, targets, val=False):
        if self.loss_type == 'ce':
            ce_loss = nn.CrossEntropyLoss(weight=self.class_weight, ignore_index=self.ignore_index).to(self.device)
            if isinstance(outputs[0], tuple) or isinstance(outputs[0], list):
                pass


class Unet_Loss(nn.modules.loss._Loss):
    def __init__(self, device, num_classes, loss_type='ce', class_weight=None, multi_scale=False):
        super(Unet_Loss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.loss_type = loss_type
        if class_weight is not None:
            self.class_weight = torch.from_numpy(np.array(class_weight)).float()
        else:
            self.class_weight = class_weight
        self.multi_scale = multi_scale

    def forward(self, outputs, targets, val=False):

        if self.loss_type == 'ce':
            ce_loss = nn.CrossEntropyLoss(weight=self.class_weight, ignore_index=255).to(self.device)
            if self.multi_scale:
                if isinstance(outputs[0],tuple) or isinstance(outputs[0],list):
                    # multi_scale_loss
                    CE_Loss = multi_scale_loss_fun(ce_loss, outputs[0], targets, self.device)
                    for i in range(1, len(outputs)):
                        _ce_loss = multi_scale_loss_fun(ce_loss, outputs[i], targets, self.device)
                        CE_Loss = CE_Loss + 0.4*_ce_loss
                else:
                    CE_Loss = multi_scale_loss_fun(ce_loss, outputs, targets, self.device)
            else:
                if isinstance(outputs,tuple) or isinstance(outputs,list):
                    CE_Loss = ce_loss(outputs[0], targets)
                    for i in range(1, len(outputs)):
                        _ce_loss = ce_loss(outputs[i], targets)
                        CE_Loss = CE_Loss + 0.4*_ce_loss
                else:
                    CE_Loss = ce_loss(outputs, targets)
        elif self.loss_type == 'large_margin':
            if val:
                ce_loss = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
            else:
                ce_loss = LargeMarginSoftmaxV1(weight=self.class_weight, ignore_index=255).to(self.device)
            if self.multi_scale:
                if isinstance(outputs[0], tuple) or isinstance(outputs[0], list):
                    # multi_scale_loss
                    CE_Loss = multi_scale_loss_fun(ce_loss, outputs[0], targets, self.device)
                    for i in range(1, len(outputs)):
                        _ce_loss = multi_scale_loss_fun(ce_loss, outputs[i], targets, self.device)
                        CE_Loss = CE_Loss + 0.4 * _ce_loss
                else:
                    CE_Loss = multi_scale_loss_fun(ce_loss, outputs, targets, self.device)
            else:
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    CE_Loss = ce_loss(outputs[0], targets)
                    for i in range(1, len(outputs)):
                        _ce_loss = ce_loss(outputs[i], targets)
                        CE_Loss = CE_Loss + 0.4 * _ce_loss
                else:
                    CE_Loss = ce_loss(outputs, targets)

        elif self.loss_type == 'ls':
            outputs = outputs.permute(0, 2, 3, 1)
            outputs = outputs.contiguous().view(-1, outputs.shape[-1])
            ce_loss = CrossEntropyLabelSmooth(11)
            CE_Loss = ce_loss(outputs, targets.view(-1))
        elif self.loss_type == 'softce_dice':
            ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=255).to(self.device)
            dice_loss = DiceLoss(mode='multiclass', ignore_index=255).to(self.device)
           
            if isinstance(outputs,tuple) or isinstance(outputs,list):
                CE_Loss = ce_loss(outputs[0], targets)
                DICE_Loss = dice_loss(outputs[0], targets)
                for i in range(1, len(outputs)):
                    _ce_loss = ce_loss(outputs[i], targets)
                    _DICE_Loss = dice_loss(outputs[i], targets)
                    CE_Loss = CE_Loss + 0.4*_ce_loss
                    DICE_Loss = LZ_Loss + 0.4*_DICE_Loss
                CE_Loss = 1.0*CE_Loss + 1.0*DICE_Loss
            else:
                CE_Loss = ce_loss(outputs, targets)
                DICE_Loss = dice_loss(outputs, targets)
                CE_Loss = 1.0*CE_Loss + 1.0*DICE_Loss
        elif self.loss_type == 'ce_with_lovasz':
            ce_loss=SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=255).to(self.device)
            if isinstance(outputs,tuple) or isinstance(outputs,list):
                CE_Loss = ce_loss(outputs[0], targets)
                LZ_Loss = lovasz_softmax(F.softmax(outputs[0], dim=1), targets, only_present=False, per_image=True, ignore=255)
                for i in range(1, len(outputs)):
                    _ce_loss = ce_loss(outputs[i], targets)
                    _LZ_Loss = lovasz_softmax(F.softmax(outputs[i], dim=1), targets, only_present=False, per_image=True, ignore=255)
                    CE_Loss = CE_Loss + 0.4*_ce_loss
                    LZ_Loss = LZ_Loss + 0.4*_LZ_Loss
                CE_Loss = 1.0*CE_Loss + 1.0*LZ_Loss
            else:
                CE_Loss = ce_loss(outputs, targets)
                LZ_Loss = lovasz_softmax(F.softmax(outputs, dim=1), targets, only_present=False, per_image=True, ignore=255)
                CE_Loss = 1.0*CE_Loss + 1.0*LZ_Loss

        return CE_Loss


class DeepLablv3_Loss(Unet_Loss):
    def __init__(self, device, num_classes, loss_type='ce', class_weight=None, aux=True, aux_weight=0.4):
        super(DeepLablv3_Loss, self).__init__(device, num_classes, loss_type, class_weight)
        self.device = device
        self.loss_type = loss_type
        if class_weight is not None:
            self.class_weight = torch.from_numpy(np.array(class_weight)).float()
        else:
            self.class_weight = class_weight
        self.num_classes = num_classes
        self.aux = aux
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        if self.aux:
            CE_Loss =  super(DeepLablv3_Loss, self).forward(outputs[0], targets)
            for i in range(1, len(outputs)):
                aux_loss = super(DeepLablv3_Loss, self).forward(outputs[i], targets)
                CE_Loss += self.aux_weight * aux_loss
        else:
            CE_Loss = super(DeepLablv3_Loss, self).forward(outputs, targets)

        # print('\r[loss] ce:%.2f\t ' % (CE_Loss.data.cpu().numpy(),), end=' ')
        return CE_Loss


def make_loss(cfg, device):
    loss = Unet_Loss(device, cfg.MODEL.N_CLASS, cfg.MODEL.LOSS_TYPE, cfg.MODEL.CLASS_WEIGHT, cfg.SOLVER.MULTI_SCALE)
    # if cfg.MODEL.NAME == 'unet' or cfg.MODEL.NAME == 'res_unet' or cfg.MODEL.NAME == 'res_unet_3plus' or cfg.MODEL.NAME == 'mf_unet' or cfg.MODEL.NAME == 'efficient_unet' or cfg.MODEL.NAME == 'dlinknet':
    #     loss = Unet_Loss(device, cfg.MODEL.N_CLASS, cfg.MODEL.LOSS_TYPE, cfg.MODEL.CLASS_WEIGHT, cfg.SOLVER.MULTI_SCALE)
    # if cfg.MODEL.NAME == 'deeplab_v3_plus' or cfg.MODEL.NAME == 'deeplab_v3':
    #     loss = Unet_Loss(device, cfg.MODEL.N_CLASS, cfg.MODEL.LOSS_TYPE, cfg.MODEL.CLASS_WEIGHT)
    return loss