# -*- coding=utf-8 -*-
#!/usr/bin/python
import sys
import os
import shutil
import numpy as np
import json
#import mmcv
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {}


def convert(xml_list, xml_dir, json_file):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
    '''
    list_fp = xml_list
    image_id=0
    # 标注基本结构
    json_dict = {"images":[],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        print(" Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)
        anns =np.load(xml_f)
        xywh=anns[:,:4]
        filename=line.replace('npy','jpg')
        # 取出图片名字
        image_id+=1
        # 图片的基本信息
        width = int(anns[0][4])
        height = int(anns[0][5])
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        # 处理每个标注的检测框
        for obj in xywh:
            # 取出检测框类别名称
            category = 'chicken'
            # 更新类别ID字典
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            xmin = int(obj[0])
            ymin = int(obj[1])
            xmax = int(obj[0]+obj[2])
            ymax = int(obj[1]+obj[3])
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            annotation = dict()
            annotation['area'] = o_width*o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # 导出到json
    #mmcv.dump(json_dict, json_file)
    #print(type(json_dict))
    json_data = json.dumps(json_dict)
    with  open(json_file, 'w') as w:
        w.write(json_data)


if __name__ == '__main__':
    root_path = '../dataset'

    if not os.path.exists(os.path.join(root_path,'coco/annotations')):
        os.makedirs(os.path.join(root_path,'coco/annotations'))
    if not os.path.exists(os.path.join(root_path, 'coco/train2017')):
        os.makedirs(os.path.join(root_path, 'coco/train2017'))
    if not os.path.exists(os.path.join(root_path, 'coco/val2017')):
        os.makedirs(os.path.join(root_path, 'coco/val2017'))
    xml_dir = '../../rare_sample/bbox_npy/' #已知的voc的标注

    xml_labels = os.listdir(xml_dir)
    np.random.shuffle(xml_labels)
    split_point = int(len(xml_labels)/10)
    # validation data
    xml_list = xml_labels[0:split_point]
    json_file = os.path.join(root_path,'coco/annotations/instances_val2017.json')
    convert(xml_list, xml_dir, json_file)
    for xml_file in xml_list:
        img_name = xml_file[:-4] + '.jpg'
        shutil.copy(os.path.join('../../rare_sample/images', img_name),
                    os.path.join(root_path, 'coco/val2017', img_name))
    # train data
    xml_list = xml_labels[split_point:]
    json_file = os.path.join(root_path,'coco/annotations/instances_train2017.json')
    convert(xml_list, xml_dir, json_file)
    for xml_file in xml_list:
        img_name = xml_file[:-4] + '.jpg'
        shutil.copy(os.path.join('../../rare_sample/images', img_name),
                    os.path.join(root_path, 'coco/train2017', img_name))