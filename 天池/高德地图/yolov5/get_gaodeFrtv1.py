import argparse

import torch.backends.cudnn as cudnn
import pandas as pd
from utils import google_utils
from utils.datasets import *
from utils.utils import *
from sklearn.cluster import KMeans
import pdb
attention_list=['car','bus','truck']
fill_null=0
def detect(save_img=False):
    features_name = ['name', 'car_cnt', 'car_sizeStd', 'car_sizeSum', 'car_size', 'car_sizeMax',
                     'car_xStd',
                     'car_xSum','car_x', 'car_xMax', 'car_xMin',
                     'car_yStd',
                     'car_ySum', 'car_y', 'car_yMax', 'car_yMin',
                     'car_disStd',
                     'car_disSum', 'car_dis','car_disMax', 'car_disMin',
                     'car_b_disStd',
                     'car_b_disSum', 'car_b_dis', 'car_b_disMax', 'car_b_disMin',
                     'car_size_disStd',
                     'car_size_disSum', 'car_size_dis', 'car_size_disMax', 'car_size_disMin',
                     'roi_car_sizeStd',
                     'roi_car_cnt', 'roi_car_sizeSum', 'roi_car_size', 'roi_car_sizeMax',
                     're_car_yStd',
                     're_car_ySum', 're_car_y', 're_car_yMax', 're_car_yMin',
                     'car_x_setStd',
                     'car_x_setSum', 'car_x_set', 'car_x_setMax', 'car_x_setMin',
                     'car_y_setStd',
                     'car_y_setSum', 'car_y_set', 'car_y_setMax', 'car_y_setMin',
                     'k_meansx', 'k_meansy', 'dis_cluster_center'
                     ]
    features_dic = {}
    fill_null = 0
    for fea in features_name:
        features_dic[fea] = []
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float().eval()  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        car_cnt = 0
        roi_area = []
        roi_car_cnt = 0
        area = []
        c_x = []
        re_c_y = []
        c_y = []
        c_dis = []
        b_dis=[]
        x_set = []
        y_set = []
        xy_set = []
        size_dis=[]
        bottom_dis=[]
        bottom_k=[]
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            img_name = path.split('/')[-1]
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                #
                for *xyxy, conf, cls in det:
                    obj_name=names[int(cls)]
                    if obj_name in attention_list:
                        h, w, _ = im0.shape
                        x1=xyxy[0]
                        y1=xyxy[1]
                        x2=xyxy[2]
                        y2=xyxy[3]
                        #
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        center_x=xywh[0]
                        center_y=xywh[1]
                        obj_w=xywh[2]
                        obj_h=xywh[3]
                        #
                        if obj_w > 0.9:
                            continue
                        if 0.25<center_x < 0.75:
                            roi_car_cnt += 1
                            roi_area.append(obj_w*obj_h)  # 目标尺寸特征
                        car_cnt += 1  # 目标数量特征
                        area.append(obj_w*obj_h)  # 目标尺寸特征
                        tmpx = abs(center_x - 0.5)
                        c_x.append(tmpx)  #
                        tmpy = abs(center_y - 0.5)
                        c_y.append(tmpy)
                        retmpy=1-center_y
                        re_c_y.append(retmpy)
                        c_dis.append(np.sqrt(tmpx * tmpx + tmpy * tmpy))
                        b_dis.append(np.sqrt(tmpx**2+retmpy**2))
                        size_dis.append(obj_w*obj_h/np.sqrt(tmpx**2+retmpy**2))
                        x_set.append(center_x)
                        y_set.append(center_y)
                        xy_set.append([center_x, center_y])
                        bottom_dis.append((1-center_y)**2)
                        bottom_k.append((1-center_y)/abs(0.5-center_x-0.001))
                if area != []:  # 对于空值使用-999来填充
                    f_areaStd = np.std(area)
                    f_areaSum = np.sum(area)
                    f_area = np.mean(area)
                    f_areaMax = np.max(area)
                else:
                    f_areaStd = fill_null
                    f_areaSum = fill_null
                    f_area = fill_null
                    f_areaMax = fill_null
                if c_x != []:
                    f_c_xStd = np.std(c_x)
                    f_c_xSum = np.sum(c_x)
                    f_c_x = np.mean(c_x)
                    f_c_xMax = np.max(c_x)
                    f_c_xMin = np.min(c_x)
                else:
                    f_c_xStd = fill_null
                    f_c_xSum = fill_null
                    f_c_x = fill_null
                    f_c_xMax = fill_null
                    f_c_xMin = fill_null
                if x_set != []:
                    f_c_x_setStd = np.std(x_set)
                    f_c_x_setSum = np.sum(x_set)
                    f_c_x_set = np.mean(x_set)
                    f_c_x_setMax = np.max(x_set)
                    f_c_x_setMin = np.min(x_set)
                else:
                    f_c_x_setStd = fill_null
                    f_c_x_setSum = fill_null
                    f_c_x_set = fill_null
                    f_c_x_setMax = fill_null
                    f_c_x_setMin = fill_null
                if y_set != []:
                    f_c_y_setStd = np.std(y_set)
                    f_c_y_setSum = np.sum(y_set)
                    f_c_y_set = np.mean(y_set)
                    f_c_y_setMax = np.max(y_set)
                    f_c_y_setMin = np.min(y_set)
                else:
                    f_c_y_setStd = fill_null
                    f_c_y_setSum = fill_null
                    f_c_y_set = fill_null
                    f_c_y_setMax = fill_null
                    f_c_y_setMin = fill_null
                if c_y != []:
                    f_c_yStd = np.std(c_y)
                    f_c_ySum = np.sum(c_y)
                    f_c_y = np.mean(c_y)
                    f_c_yMax = np.max(c_y)
                    f_c_yMin = np.min(c_y)
                else:
                    f_c_yStd = fill_null
                    f_c_ySum = fill_null
                    f_c_y = fill_null
                    f_c_yMax = fill_null
                    f_c_yMin = fill_null
                if re_c_y != []:
                    re_f_c_yStd = np.std(re_c_y)
                    re_f_c_ySum = np.sum(re_c_y)
                    re_f_c_y = np.mean(re_c_y)
                    re_f_c_yMax = np.max(re_c_y)
                    re_f_c_yMin = np.min(re_c_y)
                else:
                    re_f_c_yStd = fill_null
                    re_f_c_ySum = fill_null
                    re_f_c_y = fill_null
                    re_f_c_yMax = fill_null
                    re_f_c_yMin = fill_null
                if c_dis != []:
                    f_c_disStd = np.std(c_dis)
                    f_c_disSum = np.sum(c_dis)
                    f_c_dis = np.mean(c_dis)
                    f_c_disMax = np.max(c_dis)
                    f_c_disMin = np.min(c_dis)
                else:
                    f_c_disStd = -1
                    f_c_disSum = -1
                    f_c_dis = -1
                    f_c_disMax = -1
                    f_c_disMin = -1
                if b_dis != []:
                    f_b_disStd = np.std(b_dis)
                    f_b_disSum = np.sum(b_dis)
                    f_b_dis = np.mean(b_dis)
                    f_b_disMax = np.max(b_dis)
                    f_b_disMin = np.min(b_dis)
                else:
                    f_b_disStd = -1
                    f_b_disSum = -1
                    f_b_dis = -1
                    f_b_disMax = -1
                    f_b_disMin = -1
                if size_dis != []:
                    f_size_disStd = np.std(size_dis)
                    f_size_disSum = np.sum(size_dis)
                    f_size_dis = np.mean(size_dis)
                    f_size_disMax = np.max(size_dis)
                    f_size_disMin = np.min(size_dis)
                else:
                    f_size_disStd = -1
                    f_size_disSum = -1
                    f_size_dis = -1
                    f_size_disMax = -1
                    f_size_disMin = -1
                if roi_area != []:  # 对于空值使用-1来填充
                    f_roiareaStd = np.std(roi_area)
                    f_roiareaSum = np.sum(roi_area)
                    f_roiarea = np.mean(roi_area)
                    f_roiareaMax = np.max(roi_area)
                else:
                    f_roiareaStd = fill_null
                    f_roiareaSum = fill_null
                    f_roiarea = fill_null
                    f_roiareaMax = fill_null
                    # k-means
                if len(xy_set) > 6:  #
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(xy_set))
                    clusters = kmeans.cluster_centers_
                    k_meansx = np.mean(clusters[:, 0])
                    k_meansy = np.mean(clusters[:, 1])
                    dis_cluster_center = np.sqrt((k_meansx - 1 / 2) ** 2 + (k_meansy - 1 / 2) ** 2)
                else:
                    k_meansx = fill_null
                    k_meansy = fill_null
                    dis_cluster_center = fill_null
            else:
                f_areaStd = fill_null
                f_areaSum = fill_null
                f_area = fill_null
                f_areaMax = fill_null
                f_c_xStd = fill_null
                f_c_xSum = fill_null
                f_c_x = fill_null
                f_c_xMax = fill_null
                f_c_xMin = fill_null

                f_c_x_setStd = fill_null
                f_c_x_setSum = fill_null
                f_c_x_set = fill_null
                f_c_x_setMax = fill_null
                f_c_x_setMin = fill_null

                f_c_y_setStd = fill_null
                f_c_y_setSum = fill_null
                f_c_y_set = fill_null
                f_c_y_setMax = fill_null
                f_c_y_setMin = fill_null

                f_c_yStd = fill_null
                f_c_ySum = fill_null
                f_c_y = fill_null
                f_c_yMax = fill_null
                f_c_yMin = fill_null

                re_f_c_yStd = fill_null
                re_f_c_ySum = fill_null
                re_f_c_y = fill_null
                re_f_c_yMax = fill_null
                re_f_c_yMin = fill_null

                f_c_disStd = -1
                f_c_disSum = -1
                f_c_dis = -1
                f_c_disMax = -1
                f_c_disMin = -1

                f_b_disStd = -1
                f_b_disSum = -1
                f_b_dis = -1
                f_b_disMax = -1
                f_b_disMin = -1

                f_size_disStd = -1
                f_size_disSum = -1
                f_size_dis = -1
                f_size_disMax = -1
                f_size_disMin = -1

                f_roiareaStd = fill_null
                f_roiareaSum = fill_null
                f_roiarea = fill_null
                f_roiareaMax = fill_null
                # k-means
                k_meansx = fill_null
                k_meansy = fill_null
                dis_cluster_center = fill_null

            #
            features_dic['name'].append(img_name)
            features_dic['car_cnt'].append(car_cnt)
            features_dic['car_sizeStd'].append(f_areaStd)
            features_dic['car_sizeSum'].append(f_areaSum)
            features_dic['car_size'].append(f_area)
            features_dic['car_sizeMax'].append(f_areaMax)
            #
            features_dic['car_xStd'].append(f_c_xStd)
            features_dic['car_xSum'].append(f_c_xSum)
            features_dic['car_x'].append(f_c_x)
            features_dic['car_xMax'].append(f_c_xMax)
            features_dic['car_xMin'].append(f_c_xMin)
            #
            features_dic['car_yStd'].append(f_c_yStd)
            features_dic['car_ySum'].append(f_c_ySum)
            features_dic['car_y'].append(f_c_y)
            features_dic['car_yMax'].append(f_c_yMax)
            features_dic['car_yMin'].append(f_c_yMin)
            #
            features_dic['car_disStd'].append(f_c_disStd)
            features_dic['car_disSum'].append(f_c_disSum)
            features_dic['car_dis'].append(f_c_dis)
            features_dic['car_disMax'].append(f_c_disMax)
            features_dic['car_disMin'].append(f_c_disMin)
            #
            features_dic['car_b_disStd'].append(f_b_disStd)
            features_dic['car_b_disSum'].append(f_b_disSum)
            features_dic['car_b_dis'].append(f_b_dis)
            features_dic['car_b_disMax'].append(f_b_disMax)
            features_dic['car_b_disMin'].append(f_b_disMin)
            #
            features_dic['car_size_disStd'].append(f_size_disStd)
            features_dic['car_size_disSum'].append(f_size_disSum)
            features_dic['car_size_dis'].append(f_size_dis)
            features_dic['car_size_disMax'].append(f_size_disMax)
            features_dic['car_size_disMin'].append(f_size_disMin)
            #
            features_dic['roi_car_cnt'].append(roi_car_cnt)
            features_dic['roi_car_sizeStd'].append(f_roiareaStd)
            features_dic['roi_car_sizeSum'].append(f_roiareaSum)
            features_dic['roi_car_size'].append(f_roiarea)
            features_dic['roi_car_sizeMax'].append(f_roiareaMax)
            #
            features_dic['re_car_yStd'].append(re_f_c_yStd)
            features_dic['re_car_ySum'].append(re_f_c_ySum)
            features_dic['re_car_y'].append(re_f_c_y)
            features_dic['re_car_yMax'].append(re_f_c_yMax)
            features_dic['re_car_yMin'].append(re_f_c_yMin)
            #
            features_dic['car_x_setStd'].append(f_c_x_setStd)
            features_dic['car_x_setSum'].append(f_c_x_setSum)
            features_dic['car_x_set'].append(f_c_x_set)
            features_dic['car_x_setMax'].append(f_c_x_setMax)
            features_dic['car_x_setMin'].append(f_c_x_setMin)
            #
            features_dic['car_y_setStd'].append(f_c_y_setStd)
            features_dic['car_y_setSum'].append(f_c_y_setSum)
            features_dic['car_y_set'].append(f_c_y_set)
            features_dic['car_y_setMax'].append(f_c_y_setMax)
            features_dic['car_y_setMin'].append(f_c_y_setMin)
            #
            features_dic['k_meansx'].append(k_meansx)
            features_dic['k_meansy'].append(k_meansy)
            features_dic['dis_cluster_center'].append(dis_cluster_center)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

    print('Done. (%.3fs)' % (time.time() - t0))
    print(features_dic)
    for key in features_dic.keys():
        print(key, len(features_dic[key]))
    print(pd.DataFrame(features_dic))
    df = pd.DataFrame(features_dic)
    df.to_csv('/home/admins/qyl/Yet-Another-EfficientDet-Pytorch/featuresDetect/b_trainValFeatureYoloV51024.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5x.pt', help='model.pt path')
    parser.add_argument('--source', type=str,
                        default='./gaodeData/train',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()
