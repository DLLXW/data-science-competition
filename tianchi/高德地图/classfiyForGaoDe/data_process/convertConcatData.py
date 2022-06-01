import os
import shutil
import json
from config import Config
print("converting concat-images to train format..........")
opt=Config()
rawImgDir=opt.concat_data_dir
rawLabelDir=opt.raw_json
staus_folder=opt.trainValConcat_dir
with open(rawLabelDir) as f:
    d=json.load(f)
annos=d['annotations']
for anno in annos:
    status=anno['status']
    imgId=anno['id']
    target_folder=os.path.join(staus_folder,str(status))#
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    shutil.copy(os.path.join(rawImgDir,imgId,imgId+'.jpg'),os.path.join(target_folder,imgId+'.jpg'))

