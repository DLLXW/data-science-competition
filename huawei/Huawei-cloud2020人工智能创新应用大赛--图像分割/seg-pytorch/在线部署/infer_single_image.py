# System libs
import os
import argparse
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.dataset import InferDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg
#


def huawei_seg(imgs,segmentation_module):
    #
    cfg.merge_from_file("/home/mind/model/config/ade20k-hrnetv2-huawei.yaml")
    imgs = [imgs]
    cfg.list_test = [{'fpath_img': x} for x in imgs]
    # Dataset and Loader
    dataset_test = InferDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()
    loader=loader_test
    # Main loop
    segmentation_module.eval()
    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, 0)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, 0)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                # print(pred_tmp.shape)#torch.Size([1, 2, 1024, 1024])
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
        #
        # visualize_result(
        #     (batch_data['img_ori'], batch_data['info']),
        #     pred,
        #     cfg
        # )
        pbar.update(1)
    #
    return pred