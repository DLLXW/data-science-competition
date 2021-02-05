
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# old
old_data_dir = '/media/inch/ubuntu/data/Competition/data/'

train_image_dir = os.path.join(old_data_dir, 'train/images')
train_label_dir = os.path.join(old_data_dir, 'train/labels')
val_image_dir = os.path.join(old_data_dir, 'val/images')
val_label_dir = os.path.join(old_data_dir, 'val/labels')

# new
data_dir = '/media/inch/ubuntu/data/Competition/data/RSC_data/'

train_images_dir = os.path.join(data_dir, 'train/images')
train_labels_dir = os.path.join(data_dir, 'train/labels')
val_images_dir = os.path.join(data_dir, 'val/images')
val_labels_dir = os.path.join(data_dir, 'val/labels')

if not os.path.exists(train_images_dir):
    os.makedirs(train_images_dir)
if not os.path.exists(train_labels_dir):
    os.makedirs(train_labels_dir)
if not os.path.exists(val_images_dir):
    os.makedirs(val_images_dir)
if not os.path.exists(val_labels_dir):
    os.makedirs(val_labels_dir)

for image_dir, label_dir, new_image_dir, new_label_dir in [(train_image_dir, train_label_dir, train_images_dir, train_labels_dir), 
                                                          (val_image_dir, val_label_dir, val_images_dir, val_labels_dir)]:
    image_names = os.listdir(image_dir)
    for image_name in tqdm(image_names):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        black_image = np.zeros(image.shape)
        if False in (black_image == image):
            shutil.move(os.path.join(image_dir, image_name), os.path.join(new_image_dir, image_name))
            shutil.move(os.path.join(label_dir, image_name), os.path.join(new_label_dir, image_name))



