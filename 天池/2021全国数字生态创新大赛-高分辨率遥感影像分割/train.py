import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from utils import train_net
from dataset import RSCDataset
from dataset import train_transform, val_transform
from torch.cuda.amp import autocast
#
import segmentation_models_pytorch as smp
Image.MAX_IMAGE_PIXELS = 1000000000000000

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda")

# 准备数据集
data_dir = "./satellite_jpg/"
train_imgs_dir = os.path.join(data_dir, "img_dir/train_val/")
val_imgs_dir = os.path.join(data_dir, "img_dir/val/")

train_labels_dir = os.path.join(data_dir, "ann_dir/train_val/")
val_labels_dir = os.path.join(data_dir, "ann_dir/val/")

train_data = RSCDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
valid_data = RSCDataset(val_imgs_dir, val_labels_dir, transform=val_transform)

# 网络

class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()  
        self.model = smp.UnetPlusPlus(# UnetPlusPlus 
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x
#
model_name = 'efficientnet-b6'#xception
n_class=10
model=seg_qyl(model_name,n_class).cuda()
model= torch.nn.DataParallel(model)
# checkpoints=torch.load('outputs/efficientnet-b6-3729/ckpt/checkpoint-epoch20.pth')
# model.load_state_dict(checkpoints['state_dict'])
# 模型保存路径
save_ckpt_dir = os.path.join('./outputs/', model_name, 'ckpt')
save_log_dir = os.path.join('./outputs/', model_name)
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)

# 参数设置
param = {}

param['epochs'] = 50          # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
param['batch_size'] = 16       # 批大小
param['lr'] = 1e-2            # 学习率
param['gamma'] = 0.2          # 学习率衰减系数
param['step_size'] = 5        # 学习率衰减间隔
param['momentum'] = 0.9       # 动量
param['weight_decay'] = 5e-4    # 权重衰减
param['disp_inter'] = 1       # 显示间隔(epoch)
param['save_inter'] = 4       # 保存间隔(epoch)
param['iter_inter'] = 50     # 显示迭代间隔(batch)
param['min_inter'] = 10

param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径

# 加载权重路径（继续训练）
param['load_ckpt_dir'] = None

#
# 训练
best_model, model = train_net(param, model, train_data, valid_data)

