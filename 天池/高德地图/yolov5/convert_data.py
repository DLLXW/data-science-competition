import os
import shutil
import glob
import json
rawImgDirs=glob.glob('gaodeData/b_test_0828/*/*')
wDir='gaodeData/b_testConvert'
if not os.path.exists(wDir):
    os.makedirs(wDir)
for name in rawImgDirs:
    seq=name.split('/')[-2]
    imgId=name.split('/')[-1]

    shutil.copy(name,os.path.join(wDir,seq+'_'+imgId))

