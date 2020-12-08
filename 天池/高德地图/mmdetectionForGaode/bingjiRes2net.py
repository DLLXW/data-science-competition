import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import time
import os
import numpy as np
from torch.nn import functional as F
import json
import glob
from PIL import Image
from mmdet.apis import init_detector, inference_detector
import mmcv
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from torchvision import transforms as T
# --------------------------b7007-------------------------
import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data.distributed import DistributedSampler


import glob
from PIL import Image
from torchvision import transforms as T
import numpy as np
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'




def count(path):
    count_result = {'CT': 0, 'HX': 0, 'YD': 0, 'FB': 0}
    with open(path) as f:
        submit = json.load(f)
    submit_annos = submit['annotations']
    submit_result = []
    for i in range(len(submit_annos)):
        submit_anno = submit_annos[i]
        status = submit_anno['status']
        submit_result.append(status)
        if status == 0:
            count_result['CT'] += 1
        elif status == 1:
            count_result['HX'] += 1
        elif status == 2:
            count_result['YD'] += 1
        else:
            count_result['FB'] += 1

    return count_result

def infer_detect(model_det,test_image_path,test_json):
    detect_res = []
    with open(test_json) as f:
        d = json.load(f)
    annos = d['annotations']
    cnt_hit=0
    for anno in annos:
        imgId = anno['id']
        key_frame = anno['key_frame']
        frames=[per['frame_name'] for per in anno['frames']]
        frame_hit=0
        for frame in frames:
            img = os.path.join(test_image_path, imgId, frame)  # normal
            detections = inference_detector(model_det, img)
            detections = detections[0]
            scores = detections[:, 4]
            for j in range(len(detections)):
                score = scores[j]
                if score > 0.99:#只有当置信度高于0.999时候，才认为检测模型检测到类第四类
                    frame_hit += 1
        if frame_hit > len(frames)//2:
            print('hit...', imgId)
            cnt_hit+=1
            detect_res.append(imgId)
    print('cnt_hit...', cnt_hit)
    return detect_res

device = torch.device("cuda:0")
test_image_path = '/home/admins/qyl/gaode/raw_data/amap_traffic_b_test_0828'
test_json = '/home/admins/qyl/gaode/raw_data/amap_traffic_annotations_b_test_0828.json'

#用检测来搞第四类

CONFIG_FILE = '/home/admins/qyl/gaodemm2det//configs/res2net/cascade_rcnn_r2_101_fpn_20e_coco.py'
CHECKPOINT_PATH = '/home/admins/qyl/gaodemm2det/work_dirs/cascade_rcnn_r2_101_fpn_20e_coco/epoch_20.pth'#把刚刚那个res2net epoch_20的放在这个路径
print('begin detecting by m2det.........')
model_det = init_detector(CONFIG_FILE, CHECKPOINT_PATH)
detect_res = infer_detect(model_det,test_image_path,test_json)
print('detecting finished')
#
model = timm.create_model('swsl_resnext101_32x8d', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model_dict = torch.load('/workspace/docker/swsl_resnext101_32x8d-ca_29.pth')
model.load_state_dict(model_dict)
model = model.to(device)
model.eval()

pre_result = []
pre_name = []
pre_dict = {}
#
image_paths = sorted(glob.glob(test_image_path + '/*/*'))
for index in range(len(image_paths)):
    sample_path = image_paths[index]
    img = Image.open(sample_path)
    img = img.convert('RGB')
    img = np.array(img)
    preds_list = []
    for mode in range(2):
        transforms = get_test_transforms(mode)
        input = transforms(image=img)['image']
        input = input.unsqueeze(0)
        input = input.float()
        input = input.cuda()
        with torch.no_grad():
            output = model(input)
        _, preds = torch.max(output.data, 1)
        preds = preds.cpu().numpy().tolist()[0]
        preds_list.append(preds)
    tmp = {0: 0, 1: 0, 2: 0, 3: 0}
    for k in preds_list:
        tmp[k] += 1
    #
    most = sorted(tmp.items(), key=lambda item: item[1])[-1][0]

    pre_result.append(most)
    pre_name.append(sample_path.split('/')[-2] + '_' + sample_path.split('/')[-1])

for idx in range(len(pre_result)):
    pre_dict[pre_name[idx]] = pre_result[idx]

count_result = {'x': 0, 'y': 0, 'z': 0, "w": 0}
with open(test_json) as f:
    submit = json.load(f)
submit_annos = submit['annotations']
submit_result = []
for i in range(len(submit_annos)):
    submit_anno = submit_annos[i]
    imgId = submit_anno['id']
    frame_name = [imgId + '_' + i['frame_name'] for i in submit_anno['frames']]
    status_all = [pre_dict[i] for i in frame_name]
    status = max(status_all, key=status_all.count)
    #
    if imgId in detect_res:#如果检测模型认为赛第四类，那我们确信他就是第四类，剩下的分类来搞定，相当于取并集
        status=3
    submit['annotations'][i]['status'] = status

submit_json = '/workspace/result.json'
json_data = json.dumps(submit)
with open(submit_json, 'w') as w:
    w.write(json_data)
print(count(submit_json))
