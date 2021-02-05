"""
此代码将给定的两张图片及其标签切分成1024*1024的小图，步长选手可自行调整
随后会随机分成训练集和验证集，比例选手亦可随机调整
"""
import os
import random
import shutil
import cv2 as cv
import numpy as np

from tqdm import tqdm
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000000
TARGET_W, TARGET_H = 512, 512

def cut_images(image_name, image_path, label_path, save_dir, is_show=True):
    # 初始化路径
    image_save_dir = os.path.join(save_dir, "images/")
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    label_save_dir = os.path.join(save_dir, "labels/")
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    if is_show:
        label_show_save_dir = os.path.join(save_dir, "labels_show/")
        if not os.path.exists(label_show_save_dir):
            os.makedirs(label_show_save_dir)

    # 剪切图片
    target_w, target_h = TARGET_W, TARGET_H
    overlap = target_h // 2      # 重叠部分像素 1024//8 = 128
    stride = target_h - overlap  # 步长 1024-128 = 896
    
    image = np.asarray(Image.open(image_path))
    label = np.asarray(Image.open(label_path))
    h, w = image.shape[0], image.shape[1]  # 训练集分辨率分别是：40391*33106、34612*29810
    print("原始大小: ", w, h)
    if (w-target_w) % stride:  # 如果不能整除步长
        new_w = ((w-target_w)//stride + 1)*stride + target_w  # 更新图片大小以适应裁剪
    if (h-target_h) % stride:
        new_h = ((h-target_h)//stride + 1)*stride + target_h
    image = cv.copyMakeBorder(image, 0, new_h-h, 0, new_w-w, cv.BORDER_CONSTANT, 0)  # 扩充边界，填充0(黑色)
    label = cv.copyMakeBorder(label, 0, new_h-h, 0, new_w-w, cv.BORDER_CONSTANT, 1)  # 扩充边界，填充1(背景)
    h, w = image.shape[0], image.shape[1]
    print("填充至整数倍: ", w, h)

    def crop(cnt, crop_image, crop_label, is_show=is_show):
        _name = image_name.split(".")[0]
        image_save_path = os.path.join(image_save_dir, _name+"_"+str(cnt[0])+"_"+str(cnt[1])+".png")
        label_save_path = os.path.join(label_save_dir, _name+"_"+str(cnt[0])+"_"+str(cnt[1])+".png")
        label_show_save_path = os.path.join(label_show_save_dir, _name+"_"+str(cnt[0])+str(cnt[1])+".png")
        cv.imwrite(image_save_path, crop_image)
        cv.imwrite(label_save_path, crop_label)
        if is_show:
            cv.imwrite(label_show_save_path, crop_label*255)
    
    h, w = image.shape[0], image.shape[1]
    for i in tqdm(range((w-target_w)//stride + 1)):
        for j in range((h-target_h)//stride + 1):
            topleft_x = i*stride   # 将要裁剪成小图顶点坐标
            topleft_y = j*stride
            crop_image = image[topleft_y:topleft_y+target_h, topleft_x:topleft_x+target_w]  # 裁剪小图
            crop_label = label[topleft_y:topleft_y+target_h, topleft_x:topleft_x+target_w]
            crop((i, j), crop_image, crop_label)  # 保存裁剪后的小图
    os.remove(image_path)


def get_train_val(data_dir):
    # 初始化路径
    all_images_dir = os.path.join(data_dir, "images/")
    all_labels_dir = os.path.join(data_dir, "labels/")
    train_imgs_dir = os.path.join(data_dir, "train/images/")
    if not os.path.exists(train_imgs_dir):
        os.makedirs(train_imgs_dir)
    val_imgs_dir = os.path.join(data_dir, "val/images/")
    if not os.path.exists(val_imgs_dir):
        os.makedirs(val_imgs_dir)
    train_labels_dir = os.path.join(data_dir, "train/labels/")
    if not os.path.exists(train_labels_dir):
        os.makedirs(train_labels_dir)
    val_labels_dir = os.path.join(data_dir, "val/labels/")
    if not os.path.exists(val_labels_dir):
        os.makedirs(val_labels_dir)

    # 分配训练集和验证集
    for name in os.listdir(all_images_dir):
        image_path = os.path.join(all_images_dir, name)
        label_path = os.path.join(all_labels_dir, name)
        if os.path.exists(label_path) and os.path.exists(image_path):
            if random.randint(0, 10) < 1:
                image_save = os.path.join(val_imgs_dir, name)
                label_save = os.path.join(val_labels_dir, name)
            else:
                image_save = os.path.join(train_imgs_dir, name)
                label_save = os.path.join(train_labels_dir, name)
            shutil.move(image_path, image_save)
            shutil.move(label_path, label_save)
    total_nums = len(os.listdir(all_images_dir))
    train_nums = len(os.listdir(train_imgs_dir))
    val_nums = len(os.listdir(val_imgs_dir))
    print("all: " + str(total_nums))
    print("train: " + str(train_nums))
    print("val: " + str(val_nums))


if __name__ == "__main__":
    data_dir = "/media/inch/ubuntu/data/Competition/data"
    img_name1 = "382.png"
    img_name2 = "182.png"
    label_name1 = "382_label.png"
    label_name2 = "182_label.png"
    cut_images(img_name1, os.path.join(data_dir, img_name1), os.path.join(data_dir, label_name1), data_dir)
    cut_images(img_name2, os.path.join(data_dir, img_name2), os.path.join(data_dir, label_name2), data_dir)
    get_train_val(data_dir)