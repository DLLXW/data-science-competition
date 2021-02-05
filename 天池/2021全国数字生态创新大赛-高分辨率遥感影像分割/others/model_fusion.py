import torch
from model.efficientunet import *

fusion_name = 'b3_60_b4_79.pth'

ckpt_b3 = torch.load('trained_model/b3-epoch60.pth')
ckpt_b4 = torch.load('trained_model/b4-epoch79.pth')

checkpoint = {'b3_state_dict':{}, 'b4_state_dict':{}}

checkpoint['b3_state_dict']['state_dict'] = ckpt_b3['state_dict']
checkpoint['b4_state_dict']['state_dict'] = ckpt_b4['state_dict']

torch.save(checkpoint, fusion_name)





