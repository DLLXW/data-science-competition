import os
import json
import codecs

class_name_dic = {
  "0": "背景",
  "1": "边异常",
  "2": "角异常",
  "3": "白色点瑕疵",
  "4": "浅色块瑕疵",
  "5": "深色点块瑕疵",
  "6": "光圈瑕疵"
 }
rawImgDir='/media/ssd/compete_datasets/tile_round1_train_20201231/train_imgs/'
rawLabelDir='/media/ssd/compete_datasets/tile_round1_train_20201231/train_annos.json'
anno_dir='./yolov5/voc/Annotations/'
if not os.path.exists(anno_dir):
    os.makedirs(anno_dir)
with open(rawLabelDir) as f:
    annos=json.load(f)

#
image_ann={}
for i in range(len(annos)):
    anno=annos[i]
    name = anno['name']
    if name not in image_ann:
        image_ann[name]=[]
    image_ann[name].append(i)
#
for name in image_ann.keys():
    indexs=image_ann[name]
    height, width = annos[indexs[0]]["image_height"], annos[indexs[0]]["image_width"]
    #
    with codecs.open(anno_dir + name[:-4] + '.xml', 'w', 'utf-8') as xml:
        xml.write('<annotation>\n')
        xml.write('\t<filename>' + name + '</filename>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(3) + '</depth>\n')
        xml.write('\t</size>\n')
        cnt = 0
        for inx in indexs:
            obj = annos[inx]
            assert name == obj['name']
            bbox = obj['bbox']
            category = obj['category']
            xmin, ymin, xmax, ymax = bbox
            class_name = class_name_dic[str(category)]
            #
            xml.write('\t<object>\n')
            xml.write('\t\t<name>' + class_name + '</name>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
            cnt += 1
        assert cnt > 0
        xml.write('</annotation>')