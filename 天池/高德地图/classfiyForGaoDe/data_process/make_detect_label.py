import os
import xml
import json
import codecs
import cv2
import shutil
from config import Config

obstacles_classes = ['施工围挡', '路障', '锥桶', '告示牌1','告示牌2','施工痕迹','施工机械','工地正门']
opt=Config()
rawImgDir=opt.raw_data_dir
rawLabelDir=opt.raw_json
anno_dir='../dataset/voc2007/annotations/'
image_dir='../dataset/voc2007/JPEGImages'
if not os.path.exists(anno_dir):
    os.makedirs(anno_dir)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
with open(rawLabelDir) as f:
    d=json.load(f)
#
annos=d['annotations']
for anno in annos:
    status=anno['status']
    frames=anno['frames']
    imgId = anno['id']
    if status==3:
        for frame in frames:
            if 'obstacles' not in frame:
                continue
            obstacles=frame['obstacles']
            bboxs=[item['bbox'] for item in obstacles]
            frame_name=frame['frame_name']
            imgId_frame_name=imgId+'_'+frame_name
            image_path=os.path.join(rawImgDir, imgId, frame_name)
            shutil.copy(os.path.join(rawImgDir, imgId, frame_name), os.path.join(image_dir, imgId_frame_name))
            img = cv2.imread(image_path)
            height, width, depth = img.shape
            with codecs.open(anno_dir + imgId_frame_name[:-4] + '.xml', 'w', 'utf-8') as xml:
                xml.write('<annotation>\n')
                xml.write('\t<filename>' + imgId_frame_name + '</filename>\n')
                xml.write('\t<size>\n')
                xml.write('\t\t<width>' + str(width) + '</width>\n')
                xml.write('\t\t<height>' + str(height) + '</height>\n')
                xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
                xml.write('\t</size>\n')
                cnt = 0
                for bbox in bboxs:
                    xmin, ymin, xmax, ymax = bbox
                    class_name = 'obstacles'
                    #
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + class_name + '</name>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')
                    cnt += 1
                assert cnt > 0
                xml.write('</annotation>')