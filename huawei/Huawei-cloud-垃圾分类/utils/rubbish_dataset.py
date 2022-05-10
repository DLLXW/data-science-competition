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
class rubbishDataset(Dataset):

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
        label = np.int32(splits[1].strip(' '))
        return data.float(), label

    def __len__(self):
        return len(self.imgs)
if __name__ == '__main__':
    opt=Config()
    dataset = rubbishDataset(root=opt.train_val_data,
                      data_list_file=opt.val_list,
                      phase='test',
                      input_size=opt.input_size)

    trainloader = DataLoader(dataset, batch_size=2)
    for i, (data, label) in enumerate(trainloader):
        print(label)
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
        cv2.waitKey(0)
        break
        # dst.decode_segmap(labels.numpy()[0], plot=True)