from sklearn.utils import shuffle
from model import DogModel
from config import CFG
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import glob
from scipy import spatial
from sklearn import preprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_test_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])
class DogDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths,transforms=None):
        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        #
        return image, torch.tensor(1)
def get_image_embeddings():
    #
    model = DogModel(pretrained=False,use_fc = True,model_name=CFG.model_name).to(CFG.device)
    model.load_state_dict(torch.load(CFG.model_path))
    model.eval()
    #
    image_dataset = DogDataset(image_paths=image_paths, transforms=get_test_transforms(img_size=224))
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False
    )
    #
    embeds = []
    with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.cuda()
            label = label.cuda()
            features = model(img,label)
            #print(features.shape)
            image_embeddings = features.detach().cpu().numpy()
            embeds.append(image_embeddings)
            #
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    return image_embeddings
if __name__=="__main__":
    image_paths=glob.glob('../data/pet_biometric_challenge_2022/validation/images/*')
    image_names=[p.split('/')[-1] for p in image_paths]
    df = pd.read_csv('../data/pet_biometric_challenge_2022/validation/valid_data.csv')
    embedding_dic={}
    image_embeddings = get_image_embeddings()
    image_embeddings = preprocessing.normalize(image_embeddings,norm='l2')
    for i in range(len(image_names)):
        name=image_names[i]
        embedding_dic[name]=image_embeddings[i]
    #
    # del image_embeddings
    prediction=[]
    imageA=df['imageA'].values
    imageB=df['imageB'].values
    for i in range(len(imageA)):
        frtA=embedding_dic[imageA[i]]
        frtB=embedding_dic[imageB[i]]
        res = 1 - spatial.distance.cosine(frtA, frtB)
        print(res)
        prediction.append(res)
    df['prediction']=prediction
    df.to_csv('./submit_0509_v2.csv',index=False)