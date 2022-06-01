import torch
import torch.nn as nn
from config import Config
class wrap_xception(nn.Module):
    def __init__(self, net,opt):
        super(wrap_xception,self).__init__()

        self.xcep_net=nn.Sequential(
            net._features,
            net.pool,
        )
        self.fc_frt = nn.Linear(2048, opt.feature_dimension)
        #self.bn_frt = nn.BatchNorm1d(opt.feature_dimension)
        self.classify=nn.Linear(2048,opt.num_classes,bias=True)
    def forward(self,x):
        x=self.xcep_net(x)#[B,2048,1,1]
        frt=x.view(x.size(0), -1)#[B,2048]
        x1=self.fc_frt(frt)#[B,feature_dimension]
        #x1=self.bn_frt(x1)#this one will be the input of arc_metric
        x2=self.classify(frt)#[B,num_classes]
        return x1,x2
#the folowing try to check the wrap module
'''
if __name__ == "__main__":
    from cnn_finetune import make_model
    opt=Config()
    model  = make_model('{}'.format('xception'), num_classes=opt.num_classes,
                        pretrained=False, input_size=(opt.input_size,opt.input_size))
    x=torch.randn(4,3,opt.input_size,opt.input_size)
    model=wrap_xception(model,opt)
    model(x)
'''

