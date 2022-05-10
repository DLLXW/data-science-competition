import numpy as np
from PIL import Image
from infer_single_image import huawei_seg
import torch
import argparse
import torch.nn as nn
from infer_single_image import huawei_seg
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.config import cfg

# image_demo_dir="./demo/182_28_9.png"
# imags=Image.open(image_demo_dir)
# imags=np.array(imags)
# # data = imags.transpose(2, 0, 1)
# # print(imags.shape)
# # print(data.shape)
# imags=imags[512:1024,0:512,:]
# print(imags.shape)
# imags=Image.fromarray(np.uint8(imags))
# pred=huawei_seg(imags)
# print(pred.astype(np.int8))
# print(pred.shape)
#
# net_weight='/home/admins/qyl/huawei_compete/hrnetv2/ckpt/model_best.pth'
# print(torch.load(net_weight,map_location=lambda storage, loc: storage)["encoder"])
#---------------merge model------------------
# dic={}
# net_weight_1='/home/admins/qyl/huawei_compete/semantic-segmentation-pytorch/ckpt/ade20k-hrnetv2-c1/encoder_epoch_20.pth'
# net_weight_2='/home/admins/qyl/huawei_compete/semantic-segmentation-pytorch/ckpt/ade20k-hrnetv2-c1/decoder_epoch_20.pth'
# dic
#
# ["encoder"]=torch.load(net_weight_1)
# dic["decoder"]=torch.load(net_weight_2)
# torch.save(dic,"model_best_20.pth")
# print("merge model finished..........")
#
# Network Builders
model_path='model_best.pth'
net_encoder = ModelBuilder.build_encoder(
    arch="hrnetv2",
    fc_dim=720,
    weights=model_path)
net_decoder = ModelBuilder.build_decoder(
    arch="c1",
    fc_dim=720,
    num_class=2,
    weights=model_path,
    use_softmax=True)

crit = nn.NLLLoss(ignore_index=-1)

segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
image_demo_dir="./demo/182_28_9.png"
imags=Image.open(image_demo_dir)
imags=np.array(imags)
imags=imags[512:1024,0:512,:]
print(imags.shape)
imags=Image.fromarray(np.uint8(imags))
pred=huawei_seg(imags,segmentation_module)
print(pred)