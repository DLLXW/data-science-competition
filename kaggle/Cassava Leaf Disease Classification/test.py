from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from  torch.cuda.amp import autocast, GradScaler

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
import pydicom
import timm #from efficientnet_pytorch import EfficientNet
from vision_transformer_pytorch import VisionTransformer
from scipy.ndimage.interpolation import zoom
from sklearn.metrics import log_loss

seed = 719
train = pd.read_csv('../cassava/cassava-leaf-disease-classification/origin_train.csv')
submission = pd.read_csv('../cassava/cassava-leaf-disease-classification/sample_submission.csv')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb

class CassavaDataset(Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True
    ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']

        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])

        img  = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        # do label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

def get_inference_transforms(img_size):
    return Compose([
            Resize(img_size, img_size),
            # RandomResizedCrop(img_size, img_size),
            # Transpose(p=0.5),
            # HorizontalFlip(p=0.5),
            # VerticalFlip(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x


def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all

seed_everything(seed)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(train.shape[0]), train.label.values)

acc = []
for fold, (trn_idx, val_idx) in enumerate(folds):
    # we'll train fold 0 first
    if fold > 0:
        pass
        # break

    print('Inference fold {} started'.format(fold))


    valid_ = train.loc[val_idx,:].reset_index(drop=True)


    b4_valid_ds = CassavaDataset(valid_, '../cassava/cassava-leaf-disease-classification/train_images/', transforms=get_inference_transforms(512), output_label=False)
    vit_valid_ds = CassavaDataset(valid_, '../cassava/cassava-leaf-disease-classification/train_images/', transforms=get_inference_transforms(384), output_label=False)

    b4_val_loader = torch.utils.data.DataLoader(
        b4_valid_ds,
        batch_size=4,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
    )


    vit_val_loader = torch.utils.data.DataLoader(
        vit_valid_ds,
        batch_size=4,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
    )
    device = torch.device('cuda:2')


    val_preds = []

    vit_model = VisionTransformer.from_name('ViT-B_16', num_classes=5).to(device)
    b4_model = CassvaImgClassifier('tf_efficientnet_b4_ns', 5, pretrained=False).to(device)
    # for i in range(5):

    b4_model.load_state_dict(torch.load('2021-01-23-00-58/tf_efficientnet_b4_ns_fold_{}_best'.format(fold)))
    val_preds += [inference_one_epoch(b4_model, b4_val_loader, device)]

    vit_model.load_state_dict(torch.load('2021-01-23-20-25/vit_fold_{}_best'.format(fold)))
    val_preds += [inference_one_epoch(vit_model, vit_val_loader, device)]
#     for i in range(8, 10):
#         vit_model.load_state_dict(torch.load('../input/20210114/tf_efficientnet_b4_ns_fold_0_{}'.format(epoch)))
#         val_preds += [inference_one_epoch(vit_model, vit_val_loader, device)]


    val_preds = np.mean(val_preds, axis=0)
    print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(val_preds, axis=1)).mean()))
    acc.append((valid_.label.values==np.argmax(val_preds, axis=1)).mean())
    del b4_model, vit_model
    torch.cuda.empty_cache()

print('accuracy', np.mean(acc))

