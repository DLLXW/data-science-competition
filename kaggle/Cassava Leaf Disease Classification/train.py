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
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import timm

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
import pydicom
#from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
import fitlog
#from .vision_transformer_pytorch import VisionTransformer
from loss import BiTemperedLoss
import datetime
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--img-size', type=int, default=512)
parser.add_argument('--model', type=str, default='seresnet152d')#swsl_resnext101_32x8d tf_efficientnet_b4_ns gluon_seresnext101_64x4d 
parser.add_argument('--nfold', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--t0', type=int, default=1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-6)
parser.add_argument('--batch-size', type=int, default=12)
parser.add_argument('--accum_iter', type=int, default=2)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--random-search', type=bool, default=False)
parser.add_argument('--verbose', type=int, default=1)
# bitempered loss
parser.add_argument('--t1', type=float, default=1.)
parser.add_argument('--t2', type=float, default=1.)
parser.add_argument('--smooth-ratio', type=float, default=0.2)
parser.add_argument('--gpu', type=str, default='0,1')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

min_loss = 1e10
max_acc = 0.

save_dir = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
fitlog.set_log_dir('logs/', save_dir)
fitlog.add_hyper(args)

train = pd.read_csv('/media/limzero/qyl/leaf/dataset/train.csv')
submission = pd.read_csv('/media/limzero/qyl/leaf/dataset/sample_submission.csv')
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


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2



from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
import albumentations as A

from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return Compose([
            #Resize(args.img_size, args.img_size),
            RandomResizedCrop(args.img_size, args.img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.25),
            ShiftScaleRotate(p=0.25),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.25),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            # Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            # CenterCrop(args.img_size, args.img_size),
            Resize(args.img_size, args.img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def prepare_dataloader(df, trn_idx, val_idx, data_root='/media/limzero/qyl/leaf_data/train_images/'):
    
    #from catalyst.data.sampler import BalanceClassSampler
    
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)
        
    train_ds = CassavaDataset(train_, data_root, transforms=get_train_transforms(), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
    valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=4,
        #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        #print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs)   #output = model(input)
            #print(image_preds.shape, exam_pred.shape)

            loss = loss_fn(image_preds, image_labels)
            
            # flooding 
            # loss = (loss - 0.25).abs() + 0.25

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) %  args.accum_iter == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % args.verbose == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                
                pbar.set_description(description)
                
    if scheduler is not None and not schd_batch_update:
        scheduler.step()
        
def valid_one_epoch(fold, epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    global max_acc, min_loss

    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)   #output = model(input)
        #print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, image_labels)
        
        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]  

        if ((step + 1) % args.verbose == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)
    

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)

    acc = (image_preds_all==image_targets_all).mean()
    if loss_sum < min_loss:
        min_loss = loss_sum
        fitlog.add_best_metric({'loss': min_loss})
        fitlog.add_best_metric({'loss_epoch': epoch})
    if acc > max_acc:
        max_acc = acc
        fitlog.add_best_metric({'acc': max_acc})
        fitlog.add_best_metric({'acc_epoch': epoch})
        torch.save(model.state_dict(),'{}/{}_fold_{}_best'.format(save_dir, args.model, fold))

    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))
    
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()


class CassavaDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 fmix_params={
                     'alpha': 1., 
                     'decay_power': 3., 
                     'shape': (args.img_size, args.img_size),
                     'max_soft': True, 
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            #print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
        
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                #lam, mask = sample_mask(**self.fmix_params)
                
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
                
                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
    
                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)
                
                # mix image
                img = mask_torch*img+(1.-mask_torch)*fmix_img

                #print(mask.shape)

                #assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum()/args.img_size/args.img_size
                target = rate*target + (1.-rate)*self.labels[fmix_ix]
                #print(target, mask, img)
                #assert False
        
        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((args.img_size, args.img_size), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (args.img_size * args.img_size))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]
                
            #print('-', img.sum())
            #print(target)
            #assert False
                            
        # do label smoothing
        #print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        if model_arch[:2] == 'tf':
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, n_class)
        elif model_arch[:3] == 'vit':
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Linear(num_ftrs, n_class)
        elif model_arch[:3] == 'rep':
            num_ftrs = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(num_ftrs, n_class)
        else:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, n_class)
        #
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''
    @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x

if __name__ == '__main__':
     # for training only, need nightly build pytorch

    acc = []
    seed_everything(args.seed)
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(np.arange(train.shape[0]), train.label.values)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first
        if fold > args.nfold - 1:
            break 


        max_acc = 0.
        min_loss = 1e10
        print('Training with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, data_root='/media/limzero/qyl/leaf_data/train_images/')

        device = torch.device('cuda')
        
        if args.model == 'vit':
            model = VisionTransformer.from_pretrained('ViT-B_16', num_classes=5).to(device)
        else:
            model = CassvaImgClassifier(args.model, train.label.nunique(), pretrained=True).to(device)
            model= torch.nn.DataParallel(model)
        scaler = GradScaler()   
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25, 
        #                                                max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
        
        if args.smooth_ratio > 0.:
            criterion = BiTemperedLoss(t1=args.t1, t2=args.t2, label_smoothing=args.smooth_ratio).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device) 

        if args.model == 'vit':
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)        

        for epoch in range(args.epochs):
            train_one_epoch(epoch, model, criterion, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                valid_one_epoch(fold, epoch, model, criterion, val_loader, device, scheduler=None, schd_loss_update=False)

            torch.save(model.state_dict(),'{}/{}_fold_{}_last'.format(save_dir, args.model, fold))
            
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
        acc.append(max_acc)

    if len(acc) > 1:
        nfold = len(acc)
        fitlog.add_best_metric({str(nfold) + 'fold' : np.mean(acc)})
    fitlog.finish()
