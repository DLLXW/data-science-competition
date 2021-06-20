# [CVPR2021 PIC Challenge: 3D Face Reconstruction From Multiple 2D Images](https://tianchi.aliyun.com/competition/entrance/531885/rankingList)

**Rank 3/247**

Yuli Qu      Xi'an Jiaotong University

Miao Feng      Nankai University

Guangde Peng       University of Technology

## 0 About Dataset

The dataset consists of about 854 subjects aged from 18 to 80. Among of them, 141 subjects are captured with 25 expressions and the others with 7 expressions. Each subject is captured with 3 poses (left, frontal, right) . For each subject with different poses and expressions threre record high-resolution RGB images and 3D scans.



## 1 Method

There are two solutions to regress the key point from 2D to 3D, which are based on the *3D Morphable Model*（*3DMM*） method and the method based on only 2D regression.For this competition, we give priority to the 2D direct regression method, taking into account the convenience. At the same time, the official baseline also use resnet to directly fit 3D key points, and then predict 3 pictures  get the final key points by ensembling 3 results.But the baseline method is too simple, we have made a lot of improvements on it.

### 1.1 data preprocessing

The baseline code uses the cropped face as input, and we also crop the input image. Since the baseline does not provide the detection code, we complete the cropping based on the sota method in the field.

1)The original face image is rotated by 90 degrees, so we need to correct the image before face detection

2)We use CenterFace to detect faces, fill the edges of the detected faces into squares, and scale them to a fixed size of 256x256

3)In the previous image scale process, information loss was introduced, and then two new cropping method were proposed.

 - Using retinaface-resnet50 to detect human faces, align them, and scale them to a larger size of 600x600
 - Using retinaface-resnet50 to detect faces without scaling and cropping without alignment



### 1.2 Network structure

 Baseline uses the same model to predict the results of the key points of the face images on the left, front, and right. Due to the significant differences between the left, front, and right images, it is difficult for a single model to guarantee good results; in addition, , The baseline uses the average method to fuse the results of the 3 pictures, and cannot effectively use the information of the 3 pictures

#### Feature extraction network we designed

**Structure 1:** Use three backbones to extract the feature map of the image, and then perform model fusion at the feature map level, using a multi-level, multi-channel feature map fusion strategy

**Structure 2:** Redesign Backbone, use parallel layers of convolutional network to extract the shallower features of 3 pictures in the first few layers of backbone, merge the feature maps at the shallow layer, and then use the regression head for direct fitting

#### The regression head we designed

**Structure 1: **Only use the fully connected layer to build. Three-layer fully connected, combining non-linear activation layer and drop out layer

**Structure 2:** Considering that the key points are also a kind of sequence, we use RNN and self-attention layer to encode and decode the extracted features, and add the corresponding attention mechanism so that the relationship between the key points can be reflected.

#### loss function

We only used L1-loss to fit this task. We also tried some other Loss but failed to get an improvement.

#### The existing network structure we used

EfficientNet b3/b4/ b5/b6, SeresNet152d, Resnest200e, ResNet200d

#### Some other attempts:

- The three pictures are merged into 9 channels according to the channel dimensions. After extracting features, follow regression head
- Use separate regression heads for the x, y, and z coordinates of key points, such as xy and z
- Use the network of transformer in CV as the feature extraction network



### Experiment

1)Use AdamW as an optimizer

2)Use CosineAnnealingWarmRestarts as the learning rate scheduler

3)Use 384, 400, 416, 456, 528 and other image sizes for training

4)Set a reasonable batch_size

5)Divide the official training set into two parts, one as the validation set, and guide training according to the validation set model evaluation

### Model ensemble

- Weighted average of multiple model results
- We train the five-fold model through Group Kfold, and then use the stacking method to model ensemble. The second layer of stacking is the redesigned regression head. This method avoids manual setting of weights and can achieve better ensemble performance.