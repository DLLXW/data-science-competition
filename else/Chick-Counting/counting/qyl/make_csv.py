import pandas as pd
import numpy as np
import os
import random
import cv2
import glob
from tqdm import tqdm
data_dir='../../data/450/images'
img_paths=os.listdir(data_dir)
label_csv='../../data/train_val_df_450.csv'
df=pd.DataFrame(columns=['image_name'])
# ##-------------------写入csv-------------------
df['image_name']=img_paths
df.to_csv(label_csv,index=False)
# #-----------------------infer-----------------------
print(df)