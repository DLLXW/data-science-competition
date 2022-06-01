import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
import json
submit_result=[]

def infer_one_slice(im0,cur_x,cur_y):
    img = letterbox(im0, new_shape=opt.slice_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    #img =  img.float()  # uint8 to fp16/32
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)

    boxes = []
    max_score=0
    for i, det in enumerate(pred):  # detections per image
        # save_path = 'draw/' + image_id + '.jpg'
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                boxes.append([int(xyxy[0]+cur_x), int(xyxy[1]+cur_y), int(xyxy[2]+cur_x), int(xyxy[3]+cur_y),int(cls.item())+1,conf.item()])
                if conf.item()>max_score:
                    max_score=conf.item()
    #
    #print(max_score)
    if max_score>0.3:
        return boxes
    else:
        return []

def slice_im(image_path, sliceHeight=640, sliceWidth=640,overlap=0.01):
    #
    result_pre=[]
    image0 = cv2.imread(image_path, 1)  # color
    win_h, win_w = image0.shape[:2]
    #
    n_ims = 0
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    for y0 in range(0, image0.shape[0], dy):
        for x0 in range(0, image0.shape[1], dx):
            n_ims += 1
            #
            #这一步确保了不会出现比要切的图像小的图,其实是通过调整最后的overlop来实现的
            #举例:h=6000,w=8192.若使用640来切图,overlop:0.2*640=128,间隔就为512.所以小图的左上角坐标的纵坐标y0依次为:
            #:0,512,1024,....,5120,接下来并非为5632,因为5632+640>6000,所以y0=6000-640
            if y0 + sliceHeight > image0.shape[0]:
                y = image0.shape[0] - sliceHeight
            else:
                y = y0
            if x0 + sliceWidth > image0.shape[1]:
                x = image0.shape[1] - sliceWidth
            else:
                x = x0
            #
            # extract image
            window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
            #cv2.imwrite(outpath, window_c)
            #------对切出来的一副图像进行预测------
            slice_bbox=infer_one_slice(window_c,x,y)#返回的是这一个slice的目标集合
            if slice_bbox!=[]:
                result_pre+=slice_bbox
    return result_pre

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='../testA_imgs', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--slice_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)
    # Initialize
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=device)['model'].float().eval()  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    #
    for per_img_name in tqdm(os.listdir(opt.source)):
        image_path = os.path.join(opt.source, per_img_name)
        #
        image_result_pre = slice_im(image_path, sliceHeight=opt.slice_size, sliceWidth=opt.slice_size)
        # print(image_result_pre)
        '''
        image_result_pre:如果切图之间存在ovelap,可以经过一个NMS
        '''
        if image_result_pre != [[]]:
            for per_pre in image_result_pre:
                submit_result.append(
                    {'name': per_img_name, 'category': per_pre[4], 'bbox': per_pre[:4], 'score': per_pre[5]})
    #
    print(submit_result)
    if not os.path.exists('results/'):os.makedirs('results/')
    with open('results/resut_post.json', 'w') as fp:
        json.dump(submit_result, fp, indent=4, ensure_ascii=False)
