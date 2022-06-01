import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
class DogNet(nn.Module):
    def __init__(self, model_arch, pretrained=True,dropout=0.25):
        super().__init__()
        dict_in_feature={
                'tf_efficientnet_b4_ns':448,
                'eca_nfnet_l2':1536,
                'convnext_small':768,
                'convnext_large_in22ft1k':1536,
                'convnext_xlarge_in22ft1k':2048,
                'swsl_resnext101_32x8d':2048
                }
        self.swin = 'swin' in model_arch
        if 'tf' in model_arch or 'swsl' in model_arch:
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
        self.head=nn.Sequential(
                    nn.Linear(3*in_features, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(),
                    nn.Dropout(p = dropout),
                    #
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    #
                    nn.Linear(256, 2),
        )
        # self.head=nn.Sequential(
        #             nn.Linear(in_features, 512),
        # )
    @autocast()
    def forward(self,image_A,image_B):
        if not self.swin:
            frt_A = self.backbone(image_A)[0]
            frt_B = self.backbone(image_B)[0]#[None,channel,h,w]
            frt_A=self.pooling(frt_A).view(frt_A.shape[0], -1)
            frt_B=self.pooling(frt_B).view(frt_B.shape[0], -1)
        else:
            frt_A = self.backbone(image_A)
            frt_B = self.backbone(image_B)
        frt=torch.abs(frt_A-frt_B)
        frt=torch.cat([frt_A,frt_B,frt],axis=1)
        #print(frt.shape)
        x=self.head(frt)
        return x
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
                 model_arch='swsl_resnext101_32x8d',#convnext_large_in22ft1k
                 pretrained=False)
    print(model)
    device=torch.device('cuda')
    image_A=image_A.cuda()
    image_B=image_B.cuda()
    model.to(device)
    out=model(image_A,image_B)
    print(out.shape)