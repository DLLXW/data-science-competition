
import cv2
import glob
import os
import pandas as pd
#
'''
{'type': {'van': 0, 'truck': 1, 'car': 2, 'suv': 3, 'coach': 4, 'bus': 5, 'engineering_car': 6}, 
 'color': {'gray': 0, 'black': 1, 'indigo': 2, 'white': 3, 'red': 4, 'blue': 5, 'silvery': 6, 'brown': 7, 'gold': 8, 'yellow': 9},
 'toward': {'left': 0, 'right': 1, 'back': 2, 'front': 3}}
'''
#
columns=['type','color','toward']
# 
if __name__=="__main__":
    data_dir="../data/car/phase2_train"
    df=pd.read_csv('../data/car/phase2_train_sorted.csv')
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
        name,car_type,color,toward=row
        #
        car_type_code=[0]*len(unique_dic['type'])
        car_type_code[unique_dic['type'][car_type]]=1
        #
        color_code=[0]*len(unique_dic['color'])
        color_code[unique_dic['color'][color]]=1
        #print(clothesStyles_code)
        #
        toward_code=[0]*len(unique_dic['toward'])
        toward_code[unique_dic['toward'][toward]]=1
        #print(hairStyles_code)
        encode=car_type_code+color_code+toward_code
        target.append(encode)
    df['target']=target
    df.to_csv('./train_df.csv',index=False)
    print(df)