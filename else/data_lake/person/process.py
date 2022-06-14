
import cv2
import glob
import os
import pandas as pd
#
from PIL import Image
import torch
'''
columns=[name,upperLength,clothesStyles,hairStyles,lowerLength,lowerStyles,shoesStyles,towards,upperBlack,upperBrown,upperBlue,
        upperGreen,upperGray,upperOrange,upperPink,upperPurple,upperRed,upperWhite,upperYellow,lowerBlack,lowerBrown,lowerBlue,
        lowerGreen,lowerGray,lowerOrange,lowerPink,lowerPurple,lowerRed,lowerWhite,lowerYellow
]
'''
columns=['upperLength','clothesStyles','hairStyles','lowerLength','lowerStyles','shoesStyles','towards']

columns_color=['upperBlack','upperBrown','upperBlue','upperGreen',
        'upperGray','upperOrange','upperPink','upperPurple','upperRed','upperWhite','upperYellow',
        'lowerBlack','lowerBrown','lowerBlue','lowerGreen','lowerGray','lowerOrange','lowerPink','lowerPurple','lowerRed','lowerWhite','lowerYellow']
# 制作数据集
def make_dataset(data_paths):
    file_names=[]
    targets=[]
    for img_path in data_paths:
        target_str = img_path.split('/')[-1].split('.')[0]
        assert len(target_str) == num_char
        file_names.append(img_path.split('/')[-1])
        targets.append(target_str)
        #samples.append((img_path, target))
    #
    df["file_name"]=file_names
    df["target"]=targets
    return df
if __name__=="__main__":
    data_dir="../data/train2_new"
    df=pd.read_csv('../data/train2_new.csv')
    unique_dic={}
    for col in columns:
        tmp=df[col].unique().tolist()
        dic={}
        for i in range(len(tmp)):
            dic[tmp[i]]=i
        unique_dic[col]=dic
    print(unique_dic)
    df=df.fillna(0)
    target=[]
    for index,row in df.iterrows():
        #
        name,upperLength,clothesStyles,hairStyles,lowerLength,lowerStyles,shoesStyles,towards,upperBlack,upperBrown,upperBlue,upperGreen,upperGray,upperOrange,upperPink,upperPurple,upperRed,upperWhite,upperYellow,lowerBlack,lowerBrown,lowerBlue,lowerGreen,lowerGray,lowerOrange,lowerPink,lowerPurple,lowerRed,lowerWhite,lowerYellow=row
        upperLength_code=[0]*len(unique_dic['upperLength'])
        upperLength_code[unique_dic['upperLength'][upperLength]]=1
        #print(upperLength_code)
        #
        clothesStyles_code=[0]*len(unique_dic['clothesStyles'])
        clothesStyles_code[unique_dic['clothesStyles'][clothesStyles]]=1
        #
        hairStyles_code=[0]*len(unique_dic['hairStyles'])
        hairStyles_code[unique_dic['hairStyles'][hairStyles]]=1
        #
        lowerLength_code=[0]*len(unique_dic['lowerLength'])
        lowerLength_code[unique_dic['lowerLength'][lowerLength]]=1
        #
        lowerStyles_code=[0]*len(unique_dic['lowerStyles'])
        lowerStyles_code[unique_dic['lowerStyles'][lowerStyles]]=1
        #
        shoesStyles_code=[0]*len(unique_dic['shoesStyles'])
        shoesStyles_code[unique_dic['shoesStyles'][shoesStyles]]=1
        #
        towards_code=[0]*len(unique_dic['towards'])
        towards_code[unique_dic['towards'][towards]]=1
        #colors
        color_code=[upperBlack,upperBrown,upperBlue,upperGreen,upperGray,upperOrange,upperPink,upperPurple,upperRed,upperWhite,upperYellow,
                lowerBlack,lowerBrown,lowerBlue,lowerGreen,lowerGray,lowerOrange,lowerPink,lowerPurple,lowerRed,lowerWhite,lowerYellow]
        #
        encode=upperLength_code+clothesStyles_code+hairStyles_code+lowerLength_code+lowerStyles_code+shoesStyles_code+towards_code+color_code
        target.append(encode)
    df['target']=target
    df.to_csv('./train_df_fusai.csv',index=False)
    print(df.shape)