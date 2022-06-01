import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
import torch.nn.functional as F
class DogNet(nn.Module):
    def __init__(self, model_arch, pretrained=True,dropout=0.25):
        super().__init__()
        dict_in_feature={
                'tf_efficientnet_b4_ns':448,
                'eca_nfnet_l2':1536,
                'convnext_small':768,
                'convnext_tiny':768,
                'convnext_large_in22ft1k':1536,
                }
        self.swin = 'swin' in model_arch
        if model_arch[:2]=='tf':
            self.out_indices=[4]
        else:
            self.out_indices=[3]
        if self.swin:
            self.backbone = timm.create_model(model_arch, pretrained=pretrained,out_indices=self.out_indices)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            self.backbone = timm.create_model(model_arch, pretrained=pretrained,features_only=True,out_indices=self.out_indices)
            self.pooling=nn.AdaptiveAvgPool2d(1)
        #
        if not self.swin:
            in_features=dict_in_feature[model_arch]
        #
        # self.head=nn.Sequential(
        #             nn.Linear(in_features, 512),
        #             nn.BatchNorm1d(512),
        #             nn.LeakyReLU(),
        #             nn.Dropout(p = dropout),
        #             #
        #             nn.Linear(512, 256),
        #             nn.BatchNorm1d(256),
        #             nn.LeakyReLU(),
        #             #
        #             nn.Linear(256, 2),
        #)
    @autocast()
    def forward(self,image_A,image_B,image_C):
        if not self.swin:
            frt_A = self.backbone(image_A)[0]
            frt_B = self.backbone(image_B)[0]#[None,channel,h,w]
            frt_C = self.backbone(image_C)[0]
            frt_A=self.pooling(frt_A).view(frt_A.shape[0], -1)
            frt_B=self.pooling(frt_B).view(frt_B.shape[0], -1)
            frt_C=self.pooling(frt_C).view(frt_C.shape[0], -1)
        else:
            frt_A = self.backbone(image_A)
            frt_B = self.backbone(image_B)
            frt_C = self.backbone(image_C)
        #frt_neg_1=torch.abs(frt_A-frt_B)#neg F.pairwise_distance(embedded_x, embedded_y, 2)
        #frt_neg_2=torch.abs(frt_C-frt_B)#neg F.pairwise_distance(embedded_x, embedded_y, 2)
        # frt_pos=torch.abs(frt_A-frt_C)#pos
        # x_neg_1=self.head(frt_neg_1)
        # x_neg_2=self.head(frt_neg_2)
        # x_pos=self.head(frt_pos)
        dis_pos=F.pairwise_distance(frt_A, frt_C, 2)#
        dis_neg=F.pairwise_distance(frt_A, frt_B, 2)#
        #frt_pos=torch.abs(frt_A-frt_C)#pos
        #frt_neg=torch.abs(frt_A-frt_B)#neg F.pairwise_distance(embedded_x, embedded_y, 2)
        #x_neg=self.head(frt_neg)
        #x_pos=self.head(frt_pos)
        return dis_pos,dis_neg,frt_A,frt_B,frt_C 
if __name__=="__main__":
    #torch.Size([16, 1408, 8, 8]) torch.Size([16, 1408, 8, 8]) torch.Size([16, 1408, 8, 8])
    # model_arch='tf_efficientnet_b4_ns'#tf_efficientnet_b4_ns eca_nfnet_l2
    # model=timm.create_model(model_arch, pretrained=True,features_only=True,out_indices=[4])
    # print(model)
    # outs=model(torch.randn(2, 3, 224, 224))
    # for out in outs:
    #     print(out.shape)
    image_A=torch.randn(2, 3, 224, 224)
    image_B=torch.randn(2, 3, 224, 224)
    model  = DogNet(
                 model_arch='convnext_large_in22ft1k',#convnext_large_in22ft1k
                 pretrained=False)
    device=torch.device('cuda')
    image_A=image_A.cuda()
    image_B=image_B.cuda()
    model.to(device)
    out=model(image_A,image_B)
    print(out.shape)