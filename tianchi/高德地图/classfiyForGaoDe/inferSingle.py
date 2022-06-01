#
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import time
import os
from utils import gaodeDatasetInfer
from torch.nn import functional as F
import json
from config import Config
# some parameters
opt=Config()
#
results_dir=opt.result_dir()
rawLabelDir='raw_data/amap_traffic_annotations_test.json'
with open(rawLabelDir) as f:
    d=json.load(f)
image_datasets =gaodeDatasetInfer(opt.testConcat_dir,image_size=opt.input_size)
#
dataset_loaders = torch.utils.data.DataLoader(image_datasets,
                                                  batch_size=opt.test_batch_size,
                                                  shuffle=False, num_workers=4)
data_set_sizes = len(image_datasets)
print(data_set_sizes)
#
submit_json='submit/xcepSingle_14.json'
#net_weight='output/se_resnet50/se_resnet50Kfold5_7.pth'#'output/xception/xception_best.pth'
net_weight='outputSingle/xception/xception_14.pth'
model = torch.load(net_weight)
model.eval()
pres_dic = {}
pres_dic_map={}
for data in dataset_loaders:
    inputs, paths = data
    print(paths)
    imgSeq = paths[0].split('/')[-2]
    frame=paths[0].split('/')[-1]
    inputs = Variable(inputs.cuda())
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    pres_dic[imgSeq+'_'+frame]= preds.cpu().numpy().tolist()[0]
    if imgSeq not in pres_dic_map:
        pres_dic_map[imgSeq]=[]
    pres_dic_map[imgSeq].append(preds.cpu().numpy().tolist()[0])

#
cnt_statistic={'畅通':0,'缓行':0,'拥堵':0}
annos=d['annotations']
for i in range(len(annos)):
    anno=annos[i]
    imgId=anno['id']
    frame=anno['key_frame']
    pres=pres_dic_map[imgId]
    cnt_cls={0:0,1:0,2:0}
    for pre in pres:
        cnt_cls[pre]+=1
    most = sorted(cnt_cls.items(), key=lambda item: item[1])[-1][0]
    #status=pres_dic[imgId+'_'+frame]
    status=most
    d['annotations'][i]['status']=status
    if status==0:
        cnt_statistic['畅通']+=1
    elif status==1:
        cnt_statistic['缓行']+=1
    else:
        cnt_statistic['拥堵']+=1

json_data=json.dumps(d)
with   open(submit_json,'w') as w:
    w.write(json_data)
#
print(cnt_statistic)
    
    
    
    
