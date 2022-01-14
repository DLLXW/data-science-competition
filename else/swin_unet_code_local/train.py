import os
import torch
import torch.nn as nn
import argparse
from utils import train_net
from datasets import RSCDataset
from datasets import train_transform, val_transform
from torch.cuda.amp import autocast
from networks.vision_transformer import SwinUnet as ViT_seg
#

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config import get_config

parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')                
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')#未用到这个参数
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')#未用到这个参数
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch_size per gpu')#未用到这个参数
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', 
    metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

args = parser.parse_args()

config = get_config(args)

device = torch.device("cuda")

# 准备数据集
data_dir = "../data/"
train_imgs_dir = os.path.join(data_dir, "train_images/")
val_imgs_dir = os.path.join(data_dir, "val_images/")

train_labels_dir = os.path.join(data_dir, "train_labels/")
val_labels_dir = os.path.join(data_dir, "val_labels/")

train_data = RSCDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
valid_data = RSCDataset(val_imgs_dir, val_labels_dir, transform=val_transform)

# 网络
model_name="swin_t"
model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
model.load_from(config)
#model= torch.nn.DataParallel(model)
#model.load_state_dict(torch.load('./outputs/swin_t/ckpt/cosine_epoch20.pth'))
# x=torch.randn(1,3,224,224).cuda()
# out=model(x)
# print(out.shape)
save_ckpt_dir = os.path.join('./outputs/', model_name, 'ckpt')
save_log_dir = os.path.join('./outputs/', model_name,'ckpt')
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)

# 参数设置
param = {}

param['epochs'] = 93          # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
param['batch_size'] =16       # 批大小
param['disp_inter'] = 1       # 显示间隔(epoch)
param['iter_inter'] = 50     # 显示迭代间隔(batch)
param['min_inter'] = 10

param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0']=3  #cosine warmup的参数
param['save_epoch']={2:[5,13,29,61],3:[8,20,44,92]}
# 加载权重路径（继续训练）
param['load_ckpt_dir'] = None
#
# 训练
best_model, model = train_net(param, model, train_data, valid_data)

