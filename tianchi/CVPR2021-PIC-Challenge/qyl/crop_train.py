import cv2
import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm
import glob
from centerface import CenterFace
landmarks=True
centerface = CenterFace(landmarks=landmarks)
#
def test_image(img_path,save_path):
    #
    img_src = cv2.imread(img_path)
    img_src = np.rot90(img_src, axes=(1, 0))
    [img_h, img_w, img_c] = img_src.shape

    dets, lms = centerface(img_src, img_h, img_w, threshold=0.35)
    det = dets[0]
    boxes, score = det[:4], det[4]
    roi = [boxes[0], boxes[1], boxes[2], boxes[3]]

    w = int(roi[2] - roi[0] + 1)
    h = int(roi[3] - roi[1] + 1)
    if w>h:
        roi[1]=roi[1]-(w-h)//2
        roi[3]=roi[3]+(w-h-(w-h)//2)
    elif h>w:
        roi[0]=roi[0]-(h-w)//2
        roi[2]=roi[2]+(h-w-(h-w)//2)

    #
    src_pts = np.float32([[roi[0], roi[1]], [roi[0], roi[3]], [roi[2], roi[1]]])
    dst_pts = np.float32([[0, 0], [0, crop_dim], [crop_dim, 0]])
    tform = cv2.getAffineTransform(src_pts, dst_pts)
    im = cv2.warpAffine(img_src, tform, (crop_dim, crop_dim))
    #
    cv2.imwrite(save_path,im)


if __name__ == '__main__':
    #
    # crop_dim=256
    # train_val_txt=open('./dataset/train_val_crop.txt','w')
    # crop_dir='/home/trojanjet/3d_face/data_crop/'
    # train_src='./dataset/train_val.txt'
    # with open(train_src,'r') as f:
    #     lines=f.readlines()
    # #
    # for line in tqdm(lines):
    #     line=line.strip('\n')
    #     line=line.split(' ')
    #     tmp_path=[]
    #     for img_path in line:
    #         #
    #         tmp_path.append('/'.join(img_path.split('/')[-6:]))#相对路径写入txt，方便训练时加载图像
    #         save_path=os.path.join(crop_dir,'/'.join(img_path.split('/')[-6:-1]))
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         save_path+='/'+img_path.split('/')[-1]#crop出的人脸保存的路径
    #         #
    #         test_image(img_path,save_path)
    #     #
    #     train_val_txt.write(tmp_path[0]+' '+tmp_path[1]+' '+tmp_path[2]+'\n')
    # #
    # train_val_txt.close()
    #-----------下面这段代码是将标签也复制到crop_data目录下面去--------------
    import shutil
    img_paths=sorted(glob.glob('/home/trojanjet/3d_face/data/train/*/*/*/*/*'))
    dst_dir='/home/trojanjet/3d_face/data_crop/train/'
    for p in img_paths:
        if p.split('/')[-1] == 'nicp_106_pts.txt':
            tmp=dst_dir+'/'.join(p.split('/')[-5:-1])
            if not os.path.exists(tmp):
                os.makedirs(tmp)
            dst_path=tmp+'/nicp_106_pts.txt'
            shutil.copy(p,dst_path)