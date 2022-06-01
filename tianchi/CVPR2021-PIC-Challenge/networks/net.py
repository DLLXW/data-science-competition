import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
class landmarkRegressNet(nn.Module):
    def __init__(self, model_arch, landmarks, pretrained=False):
        super().__init__()
        self.backbone_left= timm.create_model(model_arch, pretrained=pretrained)
        num_ftrs = self.backbone_left.classifier.in_features
        self.backbone_left.classifier = nn.Identity()
        self.backbone_left.global_pool = nn.Identity()
        #
        self.backbone_mid= timm.create_model(model_arch, pretrained=pretrained)
        num_ftrs = self.backbone_mid.classifier.in_features
        self.backbone_mid.classifier = nn.Identity()
        self.backbone_mid.global_pool = nn.Identity()
        #
        self.backbone_right= timm.create_model(model_arch, pretrained=pretrained)
        num_ftrs = self.backbone_right.classifier.in_features
        self.backbone_right.classifier = nn.Identity()
        self.backbone_right.global_pool = nn.Identity()
        # if model_arch[:2] == 'tf':
        #     num_ftrs = self.model.classifier.in_features
        #     #self.model.classifier = nn.Linear(num_ftrs, landmarks)
        # elif model_arch[:3] == 'vit':
        #     num_ftrs = self.model.head.in_features
        #     #self.model.head = nn.Linear(num_ftrs, landmarks)
        # elif model_arch[:3] == 'rep':
        #     num_ftrs = self.model.head.fc.in_features
        #     #self.model.head.fc = nn.Linear(num_ftrs, landmarks)
        # else:
        #     num_ftrs = self.model.fc.in_features
        #     #self.model.fc = nn.Linear(num_ftrs, landmarks)
        self.pooling=nn.AdaptiveAvgPool2d(1)
        self.head=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512,bias=True),
            nn.Linear(512, landmarks, bias=True)
        )
        
    @autocast()
    def forward(self,image_left,image_mid,image_right):
        frt_left = self.backbone_left(image_left)
        frt_mid = self.backbone_mid(image_mid)
        frt_right = self.backbone_right(image_right)
        #
        frt=torch.mean(torch.stack([frt_left,frt_right,frt_mid]),axis=0)#[None,channel,h,w]
        frt=self.pooling(frt).view(frt.shape[0], -1)
        x=self.head(frt)
        return x
if __name__=="__main__":
    #torch.Size([16, 1408, 8, 8]) torch.Size([16, 1408, 8, 8]) torch.Size([16, 1408, 8, 8])
    image_left=torch.randn(1, 3, 256, 256)
    image_mid=torch.randn(1, 3, 256, 256)
    image_right=torch.randn(1, 3, 256, 256)
    model  = landmarkRegressNet(
                 model_arch='tf_efficientnet_b2',
                 landmarks=318,
                 pretrained=False)
    device=torch.device('cuda')
    image_left=image_left.cuda()
    image_mid=image_mid.cuda()
    image_right=image_right.cuda()
    model.to(device)
    out=model(image_left,image_mid,image_right)
    print(out.shape)