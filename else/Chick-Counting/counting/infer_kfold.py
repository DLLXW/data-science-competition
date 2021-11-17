import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # set vis gpu
thres=150
thres_min=60
thres_max=120.
if __name__ == '__main__':
    w=open('./counting.txt','w')
    w.write('filename,predicted number'+'\n')
    data_dir='../B_test500'
    datasets = Crowd(data_dir, 512, 8, is_gray=False, method='test')
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False,
                                             num_workers=1, pin_memory=False)
    #
    device = torch.device('cuda')
    epoch_minus = []
    for inputs, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.no_grad():
            pres_kfold=[]
            for fold in range(5):
                #print('Inference fold {} started'.format(fold))
                model = vgg19().to(device)
                model.eval()
                model.load_state_dict(torch.load('./ckpt/140/acc_fold_{}_best.pth'.format(fold)))
                outputs = model(inputs)
                pres=torch.sum(outputs).item()
                pres_kfold.append(pres)
                del model
                torch.cuda.empty_cache()
        pres=np.median(pres_kfold)
        # if pres<thres:
        #     pres=np.min(pres_kfold)
        # if thres_min<pres<thres_max:
        #     pres-=15
        #if pres>thres_max:
         #   pres=np.mean([thres_max,pres])
        pres_kfold=[int(cv) for cv in pres_kfold]
        print(name[0], pres_kfold ,int(pres))
        w.write(name[0]+','+str(int(pres))+'\n')
    w.close()
    #
