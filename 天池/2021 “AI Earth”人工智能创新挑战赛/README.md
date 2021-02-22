## [2021 “AI Earth”人工智能创新挑战赛](https://tianchi.aliyun.com/competition/entrance/531871/introduction)

## 赛题分析
本赛题是一个时间序列预测问题，给了两种数据(CMIP：历史模拟数据；SODA：历史观测同化数据),其中的CMIP数据有4k+，SODA只有100条。赛题任务是给定某一时刻前12个时间片的特征，然后预测未来24个月的Nino3.4 SST异常指数。
### 数据分析
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
```
如果将sst,t300,ua,va四种特征拼接起来，可以得到训练数据:[100,36,24,72,4],根据这个数据可以构造训练集，最简单的思路是将36个月分成前12，24两个部分，只使用前12个时间片的数据进行训练，后24特征部分暂不使用，只留下其对应的label当作标签。当然由于测试数据是随机点开始，所以构造训练数据时理论上选取随机点开始更好。这样构造出训练集:train_data:[100,12,24,72,4],label:[100,24].为了将train送给CNN训练，这里理解下各个维度含义，100是样本数，12是时间维度，24，72对应了2维平面位置。4可以表示四种特征(理解为四个通道)。而二维CNN的输入(N,C,H,W),这里采取将时间维度也统一到通道里面，变为[100,48,24,72].
```python
#置换维度然后合并到通道维度
....
tr_fea=tr_fea.permute(0,1,4,2,3).reshape(tr_len,-1,24,72)
val_fea=val_fea.permute(0,1,4,2,3).reshape(val_len,-1,24,72)
```
这样，将数据构造为[N,C,H,W]的格式即可套用现有的图像特征提取器，模型最后的输出应该是一个长度为24的向量。所以最后采取一个全连接层即可。对于现有的图像分类模型，最后的head改成nn.Linear(num_ftrs, 24)是最简单的方式。这里看作一种解码手段，但因为这是一个时间序列预测问题，在这里使用RNN那一套进行解码可能会取得更好的效果。
### 实验结果
- 输入:因为24，72的尺寸太小了，先将输入上采样N倍，
- efficient-b0作为backbone提取特征
- 全连接层解码
- 回归loss:mse或者l1Loss
- CMIP数据训练,10个epoch,
- 线下验证:valrmse: 0.6741 valScore: 22.6496
- 线上分数:15.5+

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
