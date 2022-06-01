import os

import PIL.Image as Image

import PIL.Image as Image
import os
import glob
from config import Config
IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 2  # 图片间隔，也就是合并成一张图后，一共有几列

# 获取图片集地址下的所有图片名称
def get_images(image_dir):

    file_name_list = sorted([name for name in os.listdir(image_dir) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item])
    if len(file_name_list)>4:
        file_name_list=file_name_list[:4-len(file_name_list)]
    if len(file_name_list)<4:
        for i in range(4-len(file_name_list)):
            file_name_list.append(file_name_list[-1])
    return file_name_list
# 定义图像拼接函数
def image_compose(image_dir=None,image_save_path=None):
    image_names=get_images(image_dir)
    print(image_dir)
    image_size=min(Image.open(os.path.join(image_dir , image_names[0])).size)
    to_image = Image.new('RGB', (IMAGE_COLUMN * image_size, IMAGE_ROW * image_size))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(os.path.join(image_dir,image_names[IMAGE_COLUMN * (y - 1) + x - 1])).resize(
                (image_size, image_size), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * image_size, (y - 1) * image_size))
    return to_image.save(image_save_path)  # 保存新图

if __name__ == '__main__':
    #
    print("begin concat four frames..........")
    opt = Config()
    data_dir = glob.glob(opt.raw_data_dir+'/*')  # amap_traffic_train_0712 when deal train dataset
    write_dir = opt.concat_data_dir  # trainValConcat when deal train dataset
    for image_dir_path in data_dir:
        image_save_path = os.path.join(write_dir, image_dir_path.split('/')[-1])
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        image_compose(image_dir=image_dir_path,
                      image_save_path=os.path.join(image_save_path, image_dir_path.split('/')[-1] + '.jpg'))
    #
    '''
    data_dir = glob.glob('../../data/amap_traffic_b_test_0828/*')# amap_traffic_train_0712 when deal train dataset
    write_dir = '../../user_data/testBConcat/' #trainValConcat when deal train dataset
    for image_dir_path in data_dir:
        image_save_path=os.path.join(write_dir,image_dir_path.split('/')[-1])
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        image_compose(image_dir=image_dir_path,image_save_path=os.path.join(image_save_path,image_dir_path.split('/')[-1]+'.jpg'))
    '''