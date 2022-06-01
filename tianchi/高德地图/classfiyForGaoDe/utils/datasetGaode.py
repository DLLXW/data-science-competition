import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import Config
class gaodeDataset(Dataset):

    def __init__(self, root, data_list_file, phase='train', input_size=640):
        self.phase = phase

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img.strip('\n')) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((input_size,input_size)),
                T.RandomHorizontalFlip(p=0.5),
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

class gaodeDatasetInfer(Dataset):

    def __init__(self,data_dir, input_size=640,transform=None):
        self.image_paths = sorted(glob.glob(data_dir+'/*/*'))
        self.transforms = transform

    def __getitem__(self, index):
        sample_path = self.image_paths[index]
        data = Image.open(sample_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(),sample_path

    def __len__(self):
        return len(self.image_paths)

if __name__ == '__main__':
    opt=Config()
    dataset = gaodeDataset(root=opt.trainValConcat_dir,
                      data_list_file=opt.val_list,
                      phase='train',
                      input_size=opt.input_size)

    trainloader = DataLoader(dataset, batch_size=2)
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
        #cv2.imshow('img', img)
        img *= np.array([0.5, 0.5, 0.5])*255
        img += np.array([0.5, 0.5, 0.5])*255
        #img += np.array([1, 1, 1])
        #img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey(10000)
        break
        # dst.decode_segmap(labels.numpy()[0], plot=True)