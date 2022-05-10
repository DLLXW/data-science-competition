
import ast
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F
from collections import OrderedDict
#from model_service.pytorch_model_service import PTServingBaseService
from model import resnet50
class garbage_classify_service(object):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.signature_key = 'predict_images'

        self.input_size = 244  # the input image size of the model

        # add the input and output key of your pb model here,
        # these keys are defined when you save a pb file
        self.input_key_1 = 'input_img'
        self.output_key_1 = 'output_score'
        self.transforms = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.model = resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 43)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device=torch.device('cpu')
        self.model.to(self.device)
        #self.model = nn.DataParallel(self.model)
        net_weight = torch.load(model_path)
        new_pre = {}
        for k, v in net_weight.items():
            name = k[7:]
            new_pre[name] = v
        #
        self.model.load_state_dict(new_pre)
        self.model.eval()

        self.label_id_name_dict = \
            {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉",
                "10": "厨余垃圾/茶叶渣",
                "11": "厨余垃圾/菜叶菜根",
                "12": "厨余垃圾/蛋壳",
                "13": "厨余垃圾/鱼骨",
                "14": "可回收物/充电宝",
                "15": "可回收物/包",
                "16": "可回收物/化妆品瓶",
                "17": "可回收物/塑料玩具",
                "18": "可回收物/塑料碗盆",
                "19": "可回收物/塑料衣架",
                "20": "可回收物/快递纸袋",
                "21": "可回收物/插头电线",
                "22": "可回收物/旧衣服",
                "23": "可回收物/易拉罐",
                "24": "可回收物/枕头",
                "25": "可回收物/毛绒玩具",
                "26": "可回收物/洗发水瓶",
                "27": "可回收物/玻璃杯",
                "28": "可回收物/皮鞋",
                "29": "可回收物/砧板",
                "30": "可回收物/纸板箱",
                "31": "可回收物/调料瓶",
                "32": "可回收物/酒瓶",
                "33": "可回收物/金属食品罐",
                "34": "可回收物/锅",
                "35": "可回收物/食用油桶",
                "36": "可回收物/饮料瓶",
                "37": "有害垃圾/干电池",
                "38": "有害垃圾/软膏",
                "39": "有害垃圾/过期药物",
                "40": "可回收物/毛巾",
                "41": "可回收物/饮料盒",
                "42": "可回收物/纸袋"
            }
    #
    def preprocess_img(self, img):
        img = img.convert('RGB')
        img = self.transforms(img)
        return img

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.preprocess_img(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data[self.input_key_1]
        img = img.unsqueeze(0)
        #
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            output = self.model(img)
            #output = F.softmax(output.data, dim=1)
            if output is not None:
                _, pred = torch.max(output.data, 1)
                pred_label = pred.cpu().numpy().tolist()[0]
                result = {'result': self.label_id_name_dict[str(pred_label)]}
            else:
                result = {'result': 'predict score is None'}
        return result

    def _postprocess(self, data):
        return data
if __name__=="__main__":
    s=garbage_classify_service(model_name='resnet50', model_path='/media/ssd/qyl/rubbish/ckpt/res50/res50_best.pth')
    img=Image.open('./img_2256.jpg')
    img = img.convert('RGB')
    img = s.preprocess_img(img)
    img={"input_img":img}
    result=s._inference(data=img)
    print(result)