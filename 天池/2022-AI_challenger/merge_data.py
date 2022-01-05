import numpy as np

data_albu=np.load('./datasets/data_2w_albu.npy')
label_albu=np.load('./datasets/label_2w_albu.npy')
data_adv=np.load('./datasets/data_adv_3w.npy')
label_adv=np.load('./datasets/label_adv_3w.npy')
#
images=np.concatenate((data_albu,data_adv),axis=0)
labels=np.concatenate((label_albu,label_adv),axis=0)
print(images.shape,labels.shape)
np.save('data.npy', images)
np.save('label.npy', labels)