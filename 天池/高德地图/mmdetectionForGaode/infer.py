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
import json
import numpy as np
#
def get_fenbi_miss():
    cnt_hit=0
    path='data/demo_test/'#normal
    save_test='data/demo_test_result/'
    if not os.path.exists(save_test):
        os.mkdir(save_test)
    debug=False
    names=os.listdir(path)
    if debug:
        img='data/demo_test/001675_3.jpg'
        pre=inference_detector(model, img)
        print(pre)
        print(len(pre))
        model.show_result(img, pre, out_file='result.jpg',score_thr=0.8)
    else:
        for index in range(len(names)):
            print(index)
            hit=0
            img=os.path.join(path,names[index])
            detections = inference_detector(model, img)
            model.show_result(img, detections, out_file=os.path.join(save_test,names[index]))
            detections=detections[0]
            scores = detections[:, 4]
            for j in range(len(detections)):
                score=scores[j]
                if score>0.7:
                    hit+=1
            if hit>0:
                cnt_hit+=1

    print('miss fengbi:{}/{}'.format(len(names)-cnt_hit,len(names)))


def get_normal_miss():
    cnt_hit=0
    path = 'data/demo_normal/'  # normal
    save_test = 'data/normal_test_result/'
    if not os.path.exists(save_test):
        os.mkdir(save_test)
    debug = False
    names = os.listdir(path)
    if debug:
        img = 'data/demo_test/001675_3.jpg'
        pre = inference_detector(model, img)
        print(pre)
        print(len(pre))
        model.show_result(img, pre, out_file='result.jpg', score_thr=0.8)
    else:
        for index in range(len(names)):
            print(index)
            hit = 0
            img = os.path.join(path, names[index])
            detections = inference_detector(model, img)
            model.show_result(img, detections, out_file=os.path.join(save_test, names[index]))
            detections = detections[0]
            scores = detections[:, 4]
            for j in range(len(detections)):
                score = scores[j]
                if score > 0.7:
                    hit += 1
            if hit > 0:
                cnt_hit += 1

    print('miss fengbi:{}/{}'.format(cnt_hit, len(names)))
def infer_online():
    json_dir='/home/admins/qyl/gaode/raw_data/amap_traffic_annotations_b_test_0828.json'
    image_dir='/home/admins/qyl/gaode/raw_data/amap_traffic_b_test_0828'
    result_detect='result_detect.txt'
    w=open(result_detect,'w')
    res_str=''
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
            res_str+=imgId+'\n'
    w.write(res_str[:-1])
    w.close()

def merge_detetct_classify():
    pass

if __name__=='__main__':
    cnt_hit = 0
    CONFIG_FILE = 'configs/res2net/cascade_rcnn_r2_101_fpn_20e_coco.py'
    CHECKPOINT_PATH = 'work_dirs/epoch_20.pth'  # 把刚刚那个res2net epoch_20的放在这个路径
    model = init_detector(CONFIG_FILE, CHECKPOINT_PATH)
    infer_online()