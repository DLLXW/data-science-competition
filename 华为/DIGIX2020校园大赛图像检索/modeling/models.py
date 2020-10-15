import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from .metric_learning import ArcMarginProduct, AddMarginProduct, AdaCos
import cirtorch



class LandmarkNet(nn.Module):

    DIVIDABLE_BY = 32

    def __init__(self,
                 n_classes,
                 model_name='resnet50',
                 pooling='GeM',
                 args_pooling: dict={},
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(LandmarkNet, self).__init__()

        self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)
        final_in_features = self.backbone.last_linear.in_features
        # HACK: work around for this issue https://github.com/Cadene/pretrained-models.pytorch/issues/120
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = getattr(cirtorch.pooling, pooling)(**args_pooling)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x

if __name__=='__main__':
    
    net=LandmarkNet(n_classes=3)
    x=torch.randn(4,3,224,224)
    x=net(x,[0,1,2,1])
    print(x)