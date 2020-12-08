#
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
from torch.nn import functional as F
import json
from cnn_finetune import make_model
from torchvision import transforms as T
import glob
from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
import os
import glob
import numpy as np
from config import Config
def infer_detect():
    CONFIG_FILE = '/home/admins/qyl/gaodemm2det/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_gaode.py'
    CHECKPOINT_PATH = '/home/admins/qyl/gaodemm2det/work_dirs/cascade_rcnn_r50_fpn_1x_gaode/latest.pth'
    model = init_detector(CONFIG_FILE, CHECKPOINT_PATH)
    json_dir='/home/admins/qyl/gaode/raw_data/amap_traffic_annotations_b_test_0828.json'
    image_dir='/home/admins/qyl/gaode/raw_data/amap_traffic_b_test_0828'
    detect_res=[]
    #result_detect='result_detect.txt'
    #w=open(result_detect,'w')
    #res_str=''
    with open(json_dir) as f:
        d = json.load(f)
    annos = d['annotations']
    for anno in annos:
        imgId = anno['id']
        frame = anno['key_frame']
        hit=0
        img = os.path.join(image_dir,imgId,frame)  # normal
        detections = inference_detector(model, img)
        detections = detections[0]
        scores = detections[:, 4]
        for j in range(len(detections)):
            score = scores[j]
            if score > 0.7:
                hit+=1
        if hit>0:
            print('hit...',imgId)
            detect_res.append(imgId)
            #res_str+=imgId+'\n'
    #w.write(res_str[:-1])
    #w.close()
    return detect_res
#
def get_test_transforms(mode):
    normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if mode == 0:
        return T.Compose([
                T.Resize((640, 640)),
                T.ToTensor(),
                normalize
            ])
    else:
        return T.Compose([
                T.RandomHorizontalFlip(p=0.8),
                T.Resize((640, 640)),
                T.ToTensor(),
                normalize
            ])

# some parameters
opt = Config()
print('begin detecting by m2det.........')
detect_res=infer_detect()
print('detecting finished')
results_json = 'result.json'
model  = make_model('{}'.format(opt.backbone), num_classes=opt.num_classes,
                        pretrained=False, input_size=(opt.input_size,opt.input_size))
device=torch.device(opt.device)
model.to(device)
model = nn.DataParallel(model)
net_weight=torch.load(opt.test_model_dir)
model.load_state_dict(net_weight)
model.eval()
pres_dic = {}
image_paths = sorted(glob.glob(opt.testConcat_dir+'/*/*'))
modes=2
print('begin classifing by xception.........')
for index in range(len(image_paths)):
    sample_path = image_paths[index]
    imgSeq = sample_path.split('/')[-2]
    data = Image.open(sample_path)
    data = data.convert('RGB')
    for mode in range(modes):
        transforms=get_test_transforms(mode)
        input = transforms(data)
        input=input.unsqueeze(0)
        input = input.float()
        input = Variable(input.cuda())
        if mode==0:
            output = model(input)/modes
        else:
            output += model(input)/modes
    _, pred = torch.max(output.data, 1)
    pres_dic[imgSeq] = pred.cpu().numpy().tolist()[0]
#
with open(opt.raw_test_json) as f:
    d = json.load(f)
#
cnt_statistic = {'畅通': 0, '缓行': 0, '拥堵': 0,'封闭':0}
annos = d['annotations']
for i in range(len(annos)):
    anno = annos[i]
    imgId = anno['id']
    if imgId in detect_res:
        status=3
    else:
        status = pres_dic[imgId]
    d['annotations'][i]['status'] = status
    if status == 0:
        cnt_statistic['畅通'] += 1
    elif status == 1:
        cnt_statistic['缓行'] += 1
    elif status == 2:
        cnt_statistic['拥堵'] += 1
    else:
        cnt_statistic['封闭'] += 1
#
json_data = json.dumps(d)
with open(results_json, 'w') as w:
    w.write(json_data)
#
print(cnt_statistic)