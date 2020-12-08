import os
import pandas as np
import cv2
from pandas.core.frame import DataFrame
image_id=[]
PredictionString=[]
def submit(pre_dir):
    names=os.listdir(pre_dir)
    for name in names:
        if name[-4:]=='.txt':
            h,w,_=cv2.imread(os.path.join(pre_dir,name[:-4]+'.jpg')).shape
            r=open(os.path.join(pre_dir,name),'r')
            for line in r.readlines():
                image_id.append(name[:-4])
                line=line.strip('\n')
                line = line.strip(' ')
                line=line.split(' ')
                line=[float(i) for i in line]
                pre='1.0'+' '+str(int(line[1]*w))+' '+str(int(line[2]*h))+' '+str(int(line[3]*w))+' '+str(int(line[4]*h))
                PredictionString.append(pre)
    sub={'image_id':image_id,'PredictionString':PredictionString}
    return DataFrame(sub)
sub=submit('inference/output/')
sub.to_csv('submition.csv',index=False)