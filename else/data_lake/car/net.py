from PIL.Image import Image
import torch
from torch import nn
import timm
from torch.cuda.amp import autocast, GradScaler
class ImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class=14, in_channels=3,pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch,in_chans=in_channels, pretrained=pretrained)
        #print(self.model)
        if model_arch[:2] == 'tf':
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, n_class)
        elif model_arch[:3] in ['vit','swi']:
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Linear(num_ftrs, n_class)
        elif model_arch[:3] in ['eca','con','rep','dm_']:
            num_ftrs = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(num_ftrs, n_class)
        else:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, n_class)
        # self.classifier = nn.Sequential(
        #         nn.Dropout(0.5),
        #         nn.Linear(num_ftrs, n_class)
        #     )
    @autocast()
    def forward(self, x):
        x = self.model(x)
        return x
#
if __name__=="__main__":
    net=ImgClassifier(model_arch='convnext_base_in22ft1k',n_class=2)
    x=torch.randn([1,3,224,224])
    net(x)