import os
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class retriDataset(Dataset):

    def __init__(self, root, data_list_file, phase='train', input_size=256,crop_size=224):
        self.phase = phase

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])


        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((input_size,input_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(30),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((input_size,input_size)),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split(',')
        img_path = splits[0]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)

class retriDatasetInfer(Dataset):

    def __init__(self,retri_dir, input_size=224,crop_size=224):
        self.retri_dir = retri_dir
        retri_data = os.listdir(self.retri_dir)
        self.img_paths = [os.path.join(self.retri_dir, each) for each in retri_data]

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

        self.transforms = T.Compose([
            T.Resize((input_size,input_size)),
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(),img_path

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    dataset = retriDataset(root='/home/admins/qyl/imageRetrival/train_data',
                      data_list_file='/home/admins/qyl/imageRetrival/train_data/label.txt',
                      phase='val',
                      input_size=256,
                      crop_size=224)

    trainloader = DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        img *= np.array([0.5, 0.5, 0.5])*255
        img += np.array([0.5, 0.5, 0.5])*255
        #img += np.array([1, 1, 1])
        #img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)