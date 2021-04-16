import os
import numpy as np
import random
import pynvml
import time

param_grid = {
        'model': 'resnet34',
        'batch_size': [8, 16, 32, 64],
        'smooth_ratio': [0.1, 0.2, 0.3, 0.4],
        't1': [0.1 * x for x in range(7, 10)],
        't2': [0.1 * x for x in range(11, 16)],
        'lr': list(np.logspace(-5, -3, base=10, num=100)),
        'weight_decay': list(np.logspace(-6, -1, base=10, num=100)),
        }
MAX_EVALS = 20

i = 0
pynvml.nvmlInit()
while i < MAX_EVALS:
    for idx in range(1, 4):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = meminfo.total
        free = meminfo.free
        if free*1.0/total > 0.5:  
            # random.seed(i)

            hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            model = 'resnet34'
            lr = hyperparameters['lr'] 
            weight_decay = hyperparameters['weight_decay']
            smooth_ratio = hyperparameters['smooth_ratio']
            t1 = hyperparameters['t1']
            t2 = hyperparameters['t2']
            batch_size = 32

            # cmd = f"nohup python train.py --random-search True --model {model} --lr {lr} --weight-decay {weight_decay} --batch-size {batch_size} --loss BiTemperedLoss --t1 {t1} --t2 {t2} --smooth-ratio {smooth_ratio} --gpu {idx} &" 
            cmd = f"nohup python eff_train.py --random-search True -t1 {t1} --t2 {t2} --smooth-ratio {smooth_ratio} --gpu {idx} &" 
            os.system(cmd)

            i += 1
            print(f'epoch {i}')

    time.sleep(30)
