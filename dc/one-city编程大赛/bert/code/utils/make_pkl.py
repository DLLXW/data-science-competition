import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
import pickle
def get_train_data(file_path, max_length=30):
    data = []
    label = []
    df = pd.read_csv(file_path)
    labels = df['label'].values.tolist()
    texts = df['text'].values.tolist()
    for line in tqdm(texts):
        ids = tokenizer.encode(line, max_length=max_length, padding='max_length', truncation=True)
        data.append(ids)
    #写入pickle
    data_pkl = {"data":data,"label":labels}
    with open("../dataset/train_data.pkl", "wb") as f:
        pickle.dump(data_pkl, f)
#
def get_test_data(file_path, max_length=30):
    data = []
    df = pd.read_csv(file_path)
    name_list=df['name'].values.tolist()
    texts = df['text'].values.tolist()
    for line in tqdm(texts):
        ids = tokenizer.encode(line, max_length=max_length, padding='max_length', truncation=True)
        data.append(ids)
    #写入pickle
    data_pkl = {"name":name_list,"data":data}
    with open("../dataset/testA_data.pkl", "wb") as f:
        pickle.dump(data_pkl, f)
if __name__=="__main__":
    pretrained = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    #
    file_path = "../usr_data/train_val_df.csv"
    get_train_data(file_path,max_length=20)
    file_path = "../usr_data/test_data_df_A.csv"
    get_test_data(file_path,max_length=20)