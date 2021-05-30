
import math
import numpy as np
import pandas as pd

import timm
import torch
from torch import nn 
import torch.nn.functional as F 
from torch.cuda.amp import autocast, GradScaler
from config import CFG
import cirtorch
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))#[11014,512]
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    @autocast()
    def forward(self, input, label):
        '''
        如何理解：cosine = F.linear(F.normalize(input), F.normalize(self.weight)).float()
            输入每一个样本被嵌入为了512维向量input:[None,512]，然后权重矩阵为weight:[11014,512]
        cosine=normalize(input)*normalize(weight):[None,11014],cosine[i,j]表示第i个样本与第j类类别中心的相似度(距离)。
        所以权重矩阵[11014,512]的物理意义可以这样描述:[i,512]表示第i个类别在高维(512)空间中的位置，也即为每一个类别的类别中心。
        输入的每一个样本被表示为了一个高维(512维)的向量，两个矩阵相乘其实可以理解为对每一个输入样本去和每一个类别的类别中心计算距离(这里计算的是归一化后的距离，也即为角度)。
        '''
        #[None,512]*[11014,512]^T -> [None,11014]
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        '''
        phi = cosine * self.cos_m - sine * self.sin_m的解释:
        根据上面内容，我们已经求出了每一个输入样本对于每一个类别中心的夹角，现在我们需要人为的加大输入和类别中心的夹角，
        假设加大的夹角为margin,则根据三角函数计算公式，cos(theta+margin)=cos(.)cos(.)-sin(.)sin(.)
        margin=0.5那么夹角约为28°
        这里的操作其实是人为的加大了输入特征嵌入和自己类别中心的夹角，但是注意的是只加大真正的类别(gt)对应的特征中心的夹角，
        其余的需要保持不变
        '''
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        #这里是将输入的label进行on-hot编码
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        #这里用one-hot乘phi:因为最后算loss只会计算真正的类别对应的位置，其余的需要保持不变
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output#, nn.CrossEntropyLoss()(output,label)


class ShopeeModel(nn.Module):

    def __init__(
        self,
        n_classes = CFG.classes,
        model_name = CFG.model_name,
        fc_dim = CFG.fc_dim,
        margin = CFG.margin,
        scale = CFG.scale,
        pooling='GeM',
        args_pooling: dict={},
        use_fc = True,
        pretrained = True):

        super(ShopeeModel,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        # in_features = self.backbone.classifier.in_features
        # self.backbone.classifier = nn.Identity()
        # self.backbone.global_pool = nn.Identity()
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        #self.pooling =  nn.AdaptiveAvgPool2d(1)
        #----------
        self.pooling = getattr(cirtorch.pooling, pooling)(**args_pooling)
        #-----------
        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            in_features = fc_dim

        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale = scale,
            margin = margin,
            easy_margin = False,
            ls_eps = 0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
    @autocast()
    def forward(self, image, label):
        features = self.extract_features(image)#
        if self.training:
            logits = self.final(features, label)
            return logits
        else:
            return features

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)#
        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.classifier(x)#
            x = self.bn(x)
        return x
if __name__=="__main__":
    net=ShopeeModel().cuda()
    x=torch.randn(4,3,384,384).cuda()
    label=torch.tensor([0,1,0,2]).cuda()
    out=net(x,label)
    print(out.shape)