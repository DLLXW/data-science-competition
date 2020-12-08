import os
import glob
import shutil
img_paths = sorted(glob.glob('/home/admins/qyl/gaode/raw_data/amap_traffic_train_0712/*/*'))
w_dir='./gaodeData/train'
if not os.path.exists(w_dir):
    os.makedirs(w_dir)
for img_path in img_paths:
    seq = img_path.split('/')[-2]
    img_name = seq + '_' + img_path.split('/')[-1]
    shutil.copy(img_path,os.path.join(w_dir,img_name))
