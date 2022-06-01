import os
import shutil
import json
from config import Config
print('convert single frame into train format....')
opt=Config()
rawImgDir=opt.raw_data_dir
rawLabelDir=opt.raw_json
staus_folder=opt.trainValSingle_dir
with open(rawLabelDir) as f:
    d=json.load(f)
annos=d['annotations']
for anno in annos:
    status=anno['status']
    imgId=anno['id']
    frame_name=[k['frame_name'] for k in anno['frames']]#图片序列
    target_folder=os.path.join(staus_folder,str(status))#不同状态的图片放到不同目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    namelist=os.listdir(os.path.join(rawImgDir,imgId))
    for name in namelist:
        shutil.copy(os.path.join(rawImgDir,imgId,name),os.path.join(target_folder,imgId+'_'+name))

