## [2021 “AI Earth”人工智能创新挑战赛](https://tianchi.aliyun.com/competition/entrance/531871/introduction)

## 2021/4/16更新
[最终方案详情](https://mp.weixin.qq.com/s/1WcmuUNZWdZ6kHcvCl9qQA) 

我们团队最终取得了线上第二的成绩
> **Rank 2/2849** 

但是因为比赛官方将线上复现训练时间限制在6hour,而我们的模型需要7h左右才能训练完毕，所以最好的成绩并未复现成功。(这里个人建议任何比赛都不应该限制模型的训练时间，限制模型融合个数或者限制模型的推理时间更为靠谱)。因为我们的方案极具创新性，所以最终赛方同意我们不参与最终决赛排名，但可以作为特邀嘉宾做一个思路分享。我们团队也决定将解决方案写成了一篇论文投会议，目前还未撰写完毕，所以暂不开源最终方案，当前只开源纯CNN线上38+的源码。





## 比赛刚开始的分析
### 实验结果
- 时间维度合并到通道维度，输入数据[Batch,12*4,24,72]
- 因为24，72的尺寸太小了，先将输入上采样
- efficient-b0作为backbone提取特征
- 全连接层解码
- 回归loss:mse或者l1Loss
- CMIP数据训练,10个epoch
- 线下验证:valrmse: 0.6741 valScore: 22.6496
- 线上分数:15.5+
## 赛题分析
本赛题是一个时间序列预测问题，给了两种数据(CMIP：历史模拟数据；SODA：历史观测同化数据),其中的CMIP数据有4k+，SODA只有100条。赛题任务是给定某一时刻前12个时间片的特征，然后预测未来24个月的Nino3.4 SST异常指数。
### 数据分析
[部分代码来自](https://mp.weixin.qq.com/s/63LPCHNo4zOA_UGDAc2xUQ)
```python
root='../enso_round1_train_20210201' 
mode='SODA'
label_path=root+'/'+mode+"_label.nc"
data_path=root+'/'+mode+"_train.nc"
nc_label= netCDF4.Dataset(label_path,'r')
tr_nc_labels=nc_label['nino'][:]
nc_SODA=netCDF4.Dataset(data_path,'r') 
nc_sst=np.array(nc_SODA['sst'][:])
nc_t300=np.array(nc_SODA['t300'][:])
nc_ua=np.array(nc_SODA['ua'][:])
nc_va=np.array(nc_SODA['va'][:])
print(nc_sst.shape)#(100, 36, 24, 72)
print(nc_t300.shape)#(100, 36, 24, 72)
print(tr_nc_labels.shape)#(100, 36)
###
tr_features = np.concatenate([nc_sst[:,:12,:,:].reshape(-1,12,24,72,1),nc_t300[:,:12,:,:].reshape(-1,12,24,72,1),nc_ua[:,:12,:,:].reshape(-1,12,24,72,1),nc_va[:,:12,:,:].reshape(-1,12,24,72,1)],axis=-1)
#
tr_features[np.isnan(tr_features)] = -0.0376#对NaN值进行均值填充
print(np.max(tr_features),np.min(tr_features),np.mean(tr_features))#17.425098419189453 -22.261333465576172
### 训练标签，取后24个
tr_labels = tr_nc_labels[:,12:] 
### 训练集验证集划分
tr_len     = int(tr_features.shape[0] * 0.8)
tr_fea     = tr_features[:tr_len,:].copy()
tr_label   = tr_labels[:tr_len,:].copy()
val_len     = tr_features.shape[0]-tr_len
val_fea     = tr_features[tr_len:,:].copy()
val_label   = tr_labels[tr_len:,:].copy()
#
tr_fea=torch.from_numpy(tr_fea)
val_fea=torch.from_numpy(val_fea)
tr_label=torch.from_numpy(tr_label)
val_label=torch.from_numpy(val_label)
#置换维度然后合并时间维度到通道维度
tr_fea=tr_fea.permute(0,1,4,2,3).reshape(tr_len,-1,24,72)
val_fea=val_fea.permute(0,1,4,2,3).reshape(val_len,-1,24,72)

```
如果将sst,t300,ua,va四种特征拼接起来，可以得到训练数据:[100,36,24,72,4],根据这个数据可以构造训练集，最简单的思路是将36个月分成前12，24两个部分，只使用前12个时间片的数据进行训练，后24特征部分暂不使用，只留下其对应的label当作标签。当然由于测试数据是随机点开始，所以构造训练数据时理论上选取随机点开始更好。这样构造出训练集:train_data:[100,12,24,72,4],label:[100,24].为了将train送给CNN训练，这里理解下各个维度含义，100是样本数，12是时间维度，24，72对应了2维平面位置。4可以表示四种特征(理解为四个通道)。而二维CNN的输入(N,C,H,W),这里采取将时间维度也统一到通道里面，变为[100,48,24,72].这样，将数据构造为[N,C,H,W]的格式即可套用现有的图像特征提取器，模型最后的输出应该是一个长度为24的向量。所以最后采取一个全连接层解码即可。但因为这是一个时间序列预测问题，在这里使用RNN那一套进行解码可能会取得更好的效果。

## 关于Docker的使用和赛题提交
[官方指导](https://tianchi.aliyun.com/competition/entrance/231759/tab/174?spm=5176.12586973.0.0.47f85330rU1xcE)

**补充个人遇到的一些问题**：
- 1 当代码用到了GPU，而又需要在docker里面debug时，则需要在原docker基础上安装nvidia docker。但还是建议在服务器debug，没有问题后再上传docker
- 2 当模型或者代码更改时，如何进行更新的问题，目前我尝试过之间docker cp ...到容器里面，然后commit，但是最后push到远端并未更新。所以一个比较粗暴的办法是根据Dockerfiles重新build,不过注意Dokerfiles里面的FROM可以直接来源于本地的Image ID

一些常用命令
```shell
sudo docker images #显示镜像
sudo docker ps -a #显示容器
sudo docker build -t registry.xxx/xxx/xxx:1.0 .#构建
sudo docker push registry.xxx/xxx/xxx:1.0:1.0 #push
```
