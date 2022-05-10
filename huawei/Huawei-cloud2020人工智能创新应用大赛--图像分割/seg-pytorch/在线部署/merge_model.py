#---------------merge model------------------
import torch
dic={}
net_weight_1='/home/admins/qyl/huawei_compete/semantic-segmentation-pytorch/ckpt/ade20k-hrnetv2-c1/encoder_epoch_2.pth'
net_weight_2='/home/admins/qyl/huawei_compete/semantic-segmentation-pytorch/ckpt/ade20k-hrnetv2-c1/decoder_epoch_2.pth'
dic["encoder"]=torch.load(net_weight_1)
dic["decoder"]=torch.load(net_weight_2)
torch.save(dic,"model_best.pth")
print("merge model finished..........")