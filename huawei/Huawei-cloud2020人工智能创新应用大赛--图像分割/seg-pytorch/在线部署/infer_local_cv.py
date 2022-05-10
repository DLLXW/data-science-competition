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

def visualize_result(data, pred, cfg):
    (img, info) = data
    colors = loadmat('data/color150.mat')['colors']
    names = {1: 'road', 2: 'background'}
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print(uniques,counts)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    #
    img_name=image_demo_dir.split('/')[-1]
    Image.fromarray(im_vis).save('demo/huaweix_epoch_3_'+img_name.replace('.jpg', '.png'))

def huawei_seg(imgs):

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-hrnetv2-huawei.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = "model_best_20.pth"
    cfg.MODEL.weights_decoder = "model_best_20.pth"
    #
    imgs = [imgs]
    cfg.list_test = [{'fpath_img': x} for x in imgs]


    torch.cuda.set_device(0)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

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

        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )
        pbar.update(1)
    #
    return pred
if __name__=="__main__":
    image_demo_dir="./demo/182_28_9.png"
    imags=Image.open(image_demo_dir)
    imags=np.array(imags)
    # data = imags.transpose(2, 0, 1)
    # print(imags.shape)
    # print(data.shape)
    #imags=imags[512:1024,0:512,:]
    print(imags.shape)
    imags=Image.fromarray(np.uint8(imags))
    pred=huawei_seg(imags)
    print(pred.astype(np.int8))
    print(pred.shape)