import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm.auto import tqdm
import shutil as sh
import torch
import sys
import glob

NMS_IOU_THR = 0.6
NMS_CONF_THR = 0.5

# WBF
best_iou_thr = 0.6
best_skip_box_thr = 0.43

# Box conf threshold
best_final_score = 0
best_score_threshold = 0

EPO = 15

WEIGHTS = 'weights/best1024.pt'

CONFIG = 'models/yolov5x.yaml'

DATA = 'data/wheat.yaml'

is_TEST = False

is_AUG = True
is_ROT = True

VALIDATE = True

PSEUDO = True

# For OOF evaluation
marking = pd.read_csv('../global-wheat-detection/train.csv')

bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    marking[column] = bboxs[:,i]
marking.drop(columns=['bbox'], inplace=True)

#>>>>>>>>>>>>>>>>>>>>>>><<<><<<<<<<<<<<<<<<<<<<<<<
