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
def findPerson(dets):
    #找到目标人，不一定是置信度最高的，但肯定是检测框面积最大的
    best_i=0
    max_area=0
    for i in range(len(dets)):
        det=dets[i]
        boxes, score = det[:4], det[4]
        roi = [boxes[0], boxes[1], boxes[2], boxes[3]]

        w = int(roi[2] - roi[0] + 1)
        h = int(roi[3] - roi[1] + 1)
        if w*h>max_area:
            best_i=i
            max_area=w*h
    return best_i

def test_image(img_path,save_path):
    #
    img_src = cv2.imread(img_path)
    img_src = np.rot90(img_src, axes=(1, 0))
    [img_h, img_w, img_c] = img_src.shape

    dets, lms = centerface(img_src, img_h, img_w, threshold=0.35)
    
    #det = dets[0]
    det=dets[findPerson(dets)]
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
    crop_dim=256
    #-----------特殊处理的图片--------------
    img_path='/home/limzero/qyl/3Dface/t.jpg'
    save_path='/home/limzero/qyl/3Dface/t1.jpg'
    test_image(img_path,save_path)