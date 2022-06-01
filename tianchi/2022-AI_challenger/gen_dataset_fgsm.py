from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from config import args_resnet, args_densenet
from utils import load_model, AverageMeter, accuracy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
'''
利用在1w张原始cifar10上训练好的Resnet模型来生成xifar10的对抗样本
代码参考pytorch 官方tutorials:

https://pytorch.org/tutorials/beginner/fgsm_tutorial.html?highlight=generative%20adversarial

'''
# Use CUDA
use_cuda = torch.cuda.is_available()
seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
#设置产生对抗样例子的参数,参数越大，人眼越能辨别与原图的差距
#epsilons = [.05, .1, .15, .2, .35]
epsilons = [.05, .15, .3]
images_glob=[]
labels_glob=[]
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform,mode="train"):
        images = np.load('data_1w.npy')
        labels = np.load('label_1w.npy')
        print("{} :".format(mode),images.shape,labels.shape)
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def main():
    #for arch in ['resnet50', 'densenet121']:
    for arch in ['resnet50']:
        if arch == 'resnet50':
            args = args_resnet
        else:
            args = args_densenet
        assert args['epochs'] <= 200
        if args['batch_size'] > 256:
            # force the batch_size to 256, and scaling the lr
            args['optimizer_hyperparameters']['lr'] *= 256/args['batch_size']
            args['batch_size'] = 256
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train,mode="train")
        valset = MyDataset(transform=transform_val,mode="eval")
        trainloader = data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)
        valloader = data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)
        # Model
        model = load_model(arch)
        model.load_state_dict(torch.load('resnet50_849.pth.tar', map_location='cpu')['state_dict'])
        model = model.cuda()
        # Test
        for epsilon in epsilons:
            train_acc,train_accs_adv = test(valloader, model,epsilon=epsilon)
            print("epsilon:{},train_acc:{},train_accs_adv{}".format(epsilon,train_acc,train_accs_adv))
            #break
# FGSM算法攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon*sign_data_grad
    # 添加剪切以维持[0,1]范围，对于标准化的情况不适用，先注释掉
    #perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image
def save_adv_sample(perturbed_inputs,soft_labels):
    '''
    将生成的对抗样本存储起来
    perturbed_inputs:[B,3,32,32]
    targets:[B,10]
    '''
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    #print(perturbed_inputs.shape)
    for i in range(perturbed_inputs.shape[0]):
        img=perturbed_inputs[i]
        soft_label=soft_labels[i].cpu().numpy()
        #print(img.shape,soft_label)
        img=img.detach().cpu().numpy()
        #print(img.shape)
        img = np.transpose(img, (1, 2, 0))
        img *= np.array(std)*255
        img += np.array(mean)*255
        img = img.astype(np.uint8)
        #cv2.imwrite('demox6.jpg',img)
        images_glob.append(img)
        labels_glob.append(soft_label)
        #break
    #

def test(trainloader, model, epsilon):
    accs = AverageMeter()
    accs_adv = AverageMeter()
    model.eval()
    #print(len(trainloader))
    for (inputs, soft_labels) in trainloader:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        inputs.requires_grad = True#获取输入梯度
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        #
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        accs.update(acc[0].item(), inputs.size(0))
        # Zero all existing gradients
        model.zero_grad()
        #
        loss.backward()
        
        #Collect datagrad
        data_grad = inputs.grad.data

        #Call FGSM Attack
        perturbed_inputs = fgsm_attack(inputs, epsilon, data_grad)
        outputs = model(perturbed_inputs)
        acc = accuracy(outputs, targets)
        accs_adv.update(acc[0].item(), inputs.size(0))
        #--
        save_adv_sample(perturbed_inputs,soft_labels)
        #break
    return  accs.avg,accs_adv.avg


if __name__ == '__main__':
    main()
    images_glob = np.array(images_glob)
    labels_glob = np.array(labels_glob)
    print(images_glob.shape,labels_glob.shape)
    #保存生成的对抗样本用于下一步训练
    np.save('data_adv_3w.npy', images_glob)
    np.save('label_adv_3w.npy', labels_glob)
