from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import pdb
from torch.nn import functional as F
from sklearn.metrics import f1_score,precision_recall_fscore_support
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import sunDatasetInfer
from albumentations import FancyPCA
# some parameters
use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_dir = 'test_input'
batch_size = 8
input_size = 224
modles_dir='output/'
results_dir='results/'
epoch=10
image_datasets =sunDatasetInfer(data_dir)
# 
dataset_loaders = torch.utils.data.DataLoader(image_datasets,
                                                  batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
data_set_sizes = len(image_datasets)
print(data_set_sizes)
# 加载
for model_name in ['ResNet','MobileNet','DenseNet', 'ShuffleNet']:
    net_dir=os.path.join(modles_dir,model_name)
    net_weight=os.path.join(net_dir,model_name + '_' + str(epoch) + '.pth')
    model= torch.load(net_weight)
    if use_gpu:
        model = model.cuda()
    # infer
    model.eval()
    pres_list=[]
    path_list=[]
    for data in dataset_loaders:
        inputs,paths = data
        paths=[name.split('.')[3].split('_')[1] for name in paths]
        inputs= Variable(inputs.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        pres_list+=preds.cpu().numpy().tolist()
        path_list+=paths
    #print(path_list)
    pres_write=[str(p+1) for p in pres_list]
    #indexs=['0'+str(100001+i)[1:] for i in range(1172)]
    w=open(os.path.join(results_dir,model_name+'_'+str(epoch)+'.txt'),'w')
    for k in range(1172):
        w.write(path_list[k]+' '+pres_write[k])
        if k!=1171:
            w.write('\n')
    #alpha,beta,betax
    re_dic={'alpha':0,'beta':0,'betax':0}
    for i in pres_list:
        if i==0:
            re_dic['alpha']+=1
        elif i==1:
            re_dic['beta']+=1
        else:
            re_dic['betax']+=1
    print(model_name,re_dic)
