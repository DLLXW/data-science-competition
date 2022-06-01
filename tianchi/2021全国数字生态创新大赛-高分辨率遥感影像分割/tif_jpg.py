import cv2.cv2 as cv
import cv2
import os
import shutil

images_dir='/media/limzero/compete_datasets/suichang_round1_train_210120/'
save_imgs='/media/limzero/compete_datasets/images'
save_masks='/media/limzero/compete_datasets/masks_new'
if not os.path.exists(save_imgs):os.makedirs(save_imgs)
if not os.path.exists(save_masks):os.makedirs(save_masks)
tif_list = [x for x in os.listdir(images_dir)]   # 获取目录中所有tif格式图像列表
for num,name in enumerate(tif_list):      # 遍历列表
    if name.endswith(".tif"):
        img = cv.imread(os.path.join(images_dir, name),-1)       #  读取列表中的tif图像
        cv.imwrite(os.path.join(save_imgs,name.split('.')[0]+".jpg"),img)    # tif 格式转 jpg 
    else:
        img =cv.imread(os.path.join(images_dir, name),cv2.IMREAD_GRAYSCALE)
        img=img-1
        cv2.imwrite(os.path.join(save_masks, name),img)
        #shutil.copy(os.path.join(images_dir, name),os.path.join(save_masks,name))
save_test='/media/limzero/compete_datasets/suichang_round1_test_partA_210120/'
save_test_dir='/media/limzero/compete_datasets/test_jpg'
if not os.path.exists(save_test_dir):os.makedirs(save_test_dir)
for name in os.listdir(save_test):
    img = cv.imread(os.path.join(save_test, name),-1)       # 
    cv.imwrite(os.path.join(save_test_dir,name.split('.')[0]+".jpg"),img)    # tif 格式转 jpg 