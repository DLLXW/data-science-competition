import math
import torch
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from transformers import AdamW
model=resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)
optimizer = AdamW(model.parameters(),lr= 2e-5)
config={"max_num_epochs":10,"iters":468}
warm_up_epochs=2
lr_milestones = [5,7]
# warm_up_with_multistep_lr
warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs else 0.1**len([m for m in lr_milestones if m <= epoch])
# warm_up_with_step_lr
gamma = 0.9; stepsize = 1
warm_up_with_step_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
    else gamma**( ((epoch - warm_up_epochs) /(config["max_num_epochs"] - warm_up_epochs))//stepsize*stepsize)
# warm_up_with_cosine_lr
warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
    else 0.5 * ( math.cos((epoch - warm_up_epochs) /(config["max_num_epochs"] - warm_up_epochs) * math.pi) + 1)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=warm_up_with_cosine_lr)


use_epoch_step=True
plt.figure()
cur_lr_list = []
cur_lr_list.append(optimizer.param_groups[-1]['lr'])
if use_epoch_step:
    for epoch in range(config["max_num_epochs"]):
        for batch in range(config["iters"]):
            '''
            这里scheduler.step(epoch + batch / iters)的理解如下,如果是一个epoch结束后再.step
            那么一个epoch内所有batch使用的都是同一个学习率,为了使得不同batch也使用不同的学习率
            则可以在这里进行.step
            '''
            #scheduler.step(epoch + batch / config["iters"])
            #scheduler.step()
            optimizer.step()
        scheduler.step()
        cur_lr=optimizer.param_groups[-1]['lr']
        cur_lr_list.append(cur_lr)
    x_list = list(range(len(cur_lr_list)))
else:
    for epoch in range(config["max_num_epochs"]):
        for batch in range(config["iters"]):
            '''
            这里scheduler.step(epoch + batch / iters)的理解如下,如果是一个epoch结束后再.step
            那么一个epoch内所有batch使用的都是同一个学习率,为了使得不同batch也使用不同的学习率
            则可以在这里进行.step
            '''
            optimizer.step()
            scheduler.step(epoch + batch / config["iters"])
            cur_lr=optimizer.param_groups[-1]['lr']
            cur_lr_list.append(cur_lr)
    x_list = list(range(len(cur_lr_list)))
plt.plot(x_list, cur_lr_list)
plt.show()