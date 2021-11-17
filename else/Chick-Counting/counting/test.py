import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # set vis gpu
thres=120
thres_min=60
thres_max=120.
if __name__ == '__main__':
    data_dir='/home/trojanjet/baidu_qyl/tianma/data/b_test500_ann/images_annos'
    datasets = Crowd(data_dir, 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False,
                                             num_workers=1, pin_memory=False)
    #
    device = torch.device('cuda')
    epoch_minus = []
    target_lst=[]
    pre_lst=[]
    acc_lst=[]
    for inputs, target, _ in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.no_grad():
            pres_kfold=[]
            for fold in [0,1,2,3,4]:
                #print('Inference fold {} started'.format(fold))
                model = vgg19().to(device)
                model.eval()
                model.load_state_dict(torch.load('./ckpt/140/acc_fold_{}_best.pth'.format(fold)))
                outputs = model(inputs)
                pre=torch.sum(outputs).item()
                pres_kfold.append(pre)
                del model
                torch.cuda.empty_cache()
        target_lst.append(target)
        pre=np.median(pres_kfold)
        pre_lst.append(pre)
        print("target {},pres {}".format(target ,pre))
        acc=1-abs(pre-target)/target
        acc_lst.append(acc)
    print(np.mean(acc_lst))
    #
