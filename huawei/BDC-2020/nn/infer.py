#
from torch.utils.data import DataLoader
import torch
from PIL import Image
from qyl_nn import MLP,GPSDataset
from torchvision import datasets
import numpy as np
#
model=MLP()#
device=torch.device('cpu')
model=model.to(device)
model.load_state_dict(torch.load('output/100.pth'))
model.eval()#
#----------------
print("-------加载模型成功----------")
test_loader = DataLoader(GPSDataset('test_data.csv',None), batch_size=1,shuffle=False)
pres=[]
for data in test_loader:
    data = torch.tensor(data, dtype=torch.float32)
    data = data.to(device)
    output = model(data)
    pres.append(output[0].item())
answers=np.array(pres)
##观察预测结果的分布
re_day=answers/24
name_cnt={}#{订单号：对应的事件记录条数}
tmp=[]
print("预测的最长时间:%s day"%(np.max(re_day)))
print("预测的最短时间:%s day"%np.min(re_day))
print("预测的平均时间:%s day"%np.mean(re_day))
#画出分布直方图
import pylab as plt
bins = np.linspace(int(min(re_day)),int(max(re_day)),int(max(re_day)))
plt.hist(re_day,bins)
plt.xlabel('ETA of orders ')
plt.ylabel('Number of orders')
plt.show()
