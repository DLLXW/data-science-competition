# -*- coding: utf-8 -*-
"""
@Time ： 2021/1/8 下午5:01
@Auth ： https://github.com/Wakinguup/Underwater_detection/blob/master/code/draw_bbox.ipynb
@File ：draw_bbox.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import json
import cv2
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
#
def get(root, name):
    vars = root.findall(name)
    return vars
def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars
def deal_xml(xml_f):
    tree = ET.parse(xml_f)
    root = tree.getroot()
    object_list=[]
    # 处理每个标注的检测框
    for obj in get(root, 'object'):
        # 取出检测框类别名称
        category = get_and_check(obj, 'name', 1).text
        # 更新类别ID字典
        bndbox = get_and_check(obj, 'bndbox', 1)
        xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
        ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
        xmax = int(get_and_check(bndbox, 'xmax', 1).text)
        ymax = int(get_and_check(bndbox, 'ymax', 1).text)
        assert (xmax > xmin)
        assert (ymax > ymin)
        o_width = abs(xmax - xmin)
        o_height = abs(ymax - ymin)
        obj_info=[xmin,ymin,xmax,ymax,category]
        object_list.append(obj_info)
    return object_list
#
def draw_voc():
    ann_dir = './slice/annotations'
    image_dir = './slice/JPEGImages'
    save_path = './slice/val_with_bbox/'  # the path of saveing image with annotated bboxes
    if not os.path.exists(save_path): os.makedirs(save_path)

    for ann_name in tqdm(os.listdir(ann_dir)):
        ann_path=os.path.join(ann_dir,ann_name)
        object_list=deal_xml(ann_path)
        img = cv2.imread(os.path.join(image_dir , ann_name[:-4]+'.jpg'))
        for obj in object_list:
            x1 = obj[0]
            y1 = obj[1]
            x2 = obj[2]
            y2 = obj[3]
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 8)
        cv2.imwrite(os.path.join(save_path , ann_name[:-4]+'.jpg'),img)
def draw_coco():

    ann_path = 'coco/annotations/instances_val2017.json' # annotation json
    img_path = 'coco/val2017/'
    save_path = 'coco/val2017_with_bbox/' # the path of saveing image with annotated bboxes
    #
    if not os.path.exists(save_path):os.makedirs(save_path)
    with open(ann_path,'r') as f:
        ann = json.load(f)
    #
    # for ann_img in tqdm(ann['images']):
    #     img = cv2.imread(img_path + ann_img['file_name'])
    #     img_id = ann_img['id']
    #     for ann_ann in ann['annotations']:
    #         if ann_ann['image_id'] == img_id:
    #             x1 = ann_ann['bbox'][0]
    #             y1 = ann_ann['bbox'][1]
    #             x2 = ann_ann['bbox'][0] + ann_ann['bbox'][2]
    #             y2 = ann_ann['bbox'][1] + ann_ann['bbox'][3]
    #             img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 8)
    #     cv2.imwrite(save_path + ann_img['file_name'], img)
    # #
    aug_anns = ann
    print("The augmentation image number: %d" % len(aug_anns['images']))
    print("The augmentation annotation number: %d" % len(aug_anns['annotations']))
    print("")
    class_freq_dict = {}

    # init class_fre_dict
    for cls in aug_anns['categories']:
        class_freq_dict[cls['id']] = 0

    # count the instance number of each class
    for ann in aug_anns['annotations']:
        class_freq_dict[ann['category_id']] += 1

    # print out class frequency
    print("The instance number of each class:")
    for cls_id in list(class_freq_dict.keys()):
        for cat in aug_anns['categories']:
            if cat['id'] == cls_id:
                print(cat['name'], ': ', class_freq_dict[cls_id])

#
if __name__=="__main__":
    draw_voc()