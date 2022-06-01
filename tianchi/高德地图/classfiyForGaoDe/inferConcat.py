#
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
from utils import gaodeDatasetInfer
from torch.nn import functional as F
import json
from config import Config
from cnn_finetune import make_model
from torchvision import transforms as T
import glob
from PIL import Image
# some parameters
opt = Config()
#
def get_test_transforms(mode):
    normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if mode == 0:
        return T.Compose([
                T.Resize((800, 800)),
                T.ToTensor(),
                normalize
            ])
    else:
        return T.Compose([
                T.RandomHorizontalFlip(p=0.8),
                T.Resize((800, 800)),
                T.ToTensor(),
                normalize
            ])
results_json = opt.result_json


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
print(image_paths)
modes=1
for index in range(len(image_paths)):
    sample_path = image_paths[index]
    imgSeq = sample_path.split('/')[-2]
    print(imgSeq)
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
    # frame=anno['key_frame']
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




