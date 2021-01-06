## [2021广东工业智造创新大赛—智能算法赛](https://tianchi.aliyun.com/competition/entrance/531846/introduction)

大赛数据覆盖到了瓷砖产线所有常见瑕疵，包括粉团、角裂、滴釉、断墨、滴墨、B孔、落脏、边裂、缺角、砖渣、白边等。初始的标注格式较为零散，为了便于快速构建模型同时更加方便的使用mmdetection这种开源框架。这里给出将原始数据转化为voc,coco标准格式的代码。只需要更改数据读取和保存路径即可运行!

## convert_to_voc.py

该脚本将原始的标注文件转化为voc格式的数据，方便大多数开源框架的训练

## voc_to_coco.py

将上面转换好的voc数据转换为coco数据集的标准格式,方便快速构建训练pipeline

## yolo系列的数据格式

[参见本人的另一个repo](https://github.com/DLLXW/objectDetectionDatasets)



今天刚参见比赛，只是对数据进行了一个简单的转换和处理,后续如果条件合适,会在本仓库继续开源本比赛的baseline