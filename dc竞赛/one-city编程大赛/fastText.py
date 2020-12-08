import pandas as pd
from pandas import DataFrame
import numpy as np
import jieba
import fasttext
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import random

random.seed(2020)

def get_list_list(list_1, list_2):
    return [list_1[i] for i in list_2]


def make_fold_txt():
    #
    save_data = 'fastTextData_kfold/'
    if not os.path.exists(save_data):
        os.mkdir(save_data)
    train_txt1 = open(save_data + 'train_1.txt', 'w', encoding='utf-8')
    val_txt1 = open(save_data + 'val_1.txt', 'w', encoding='utf-8')
    train_txt2 = open(save_data + 'train_2.txt', 'w', encoding='utf-8')
    val_txt2 = open(save_data + 'val_2.txt', 'w', encoding='utf-8')
    train_txt3 = open(save_data + 'train_3.txt', 'w', encoding='utf-8')
    val_txt3 = open(save_data + 'val_3.txt', 'w', encoding='utf-8')
    train_txt4 = open(save_data + 'train_4.txt', 'w', encoding='utf-8')
    val_txt4 = open(save_data + 'val_4.txt', 'w', encoding='utf-8')
    train_txt5 = open(save_data + 'train_5.txt', 'w', encoding='utf-8')
    val_txt5 = open(save_data + 'val_5.txt', 'w', encoding='utf-8')
    #
    index = 0
    train_all = train_data['text'].values.tolist()
    kind = train_data['label'].values.tolist()
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20)
    fold = 0
    for train_idx, test_idx in sk.split(train_all, kind):
        fold += 1
        print(fold)
        X_train = get_list_list(train_all, train_idx)
        y_train = get_list_list(kind, train_idx)
        X_test = get_list_list(train_all, test_idx)
        y_test = get_list_list(kind, test_idx)

        for tra_inx in range(len(X_train)):
            text = X_train[tra_inx]
            label = y_train[tra_inx]
            seg_text = jieba.cut(text)
            out_str = " ".join(seg_text)
            if fold == 1:
                train_txt1.write('%s __label__%d\n' % (out_str, int(label)))
            elif fold == 2:
                train_txt2.write('%s __label__%d\n' % (out_str, int(label)))
            elif fold == 3:
                train_txt3.write('%s __label__%d\n' % (out_str, int(label)))
            elif fold == 4:
                train_txt4.write('%s __label__%d\n' % (out_str, int(label)))
            else:
                train_txt5.write('%s __label__%d\n' % (out_str, int(label)))
        for tes_inx in range(len(X_test)):
            text = X_test[tes_inx]
            label = y_test[tes_inx]
            seg_text = jieba.cut(text)
            out_str = " ".join(seg_text)
            if fold == 1:
                val_txt1.write('%s __label__%d\n' % (out_str, int(label)))
            elif fold == 2:
                val_txt2.write('%s __label__%d\n' % (out_str, int(label)))
            elif fold == 3:
                val_txt3.write('%s __label__%d\n' % (out_str, int(label)))
            elif fold == 4:
                val_txt4.write('%s __label__%d\n' % (out_str, int(label)))
            else:
                val_txt5.write('%s __label__%d\n' % (out_str, int(label)))
    #
    train_txt1.close()
    val_txt1.close()
    train_txt2.close()
    val_txt2.close()
    train_txt3.close()
    val_txt3.close()
    train_txt4.close()
    val_txt4.close()
    train_txt5.close()
    val_txt5.close()

if __name__=="__main__":
    answer_train=pd.read_csv('data/answer_train.csv')
    encode_dic={}
    cate=answer_train.label.unique()
    for i in range(len(cate)):
        encode_dic[cate[i]]=i
    answer_train['label']=answer_train['label'].map(encode_dic)
    filename_label={}
    for index,filename,label in answer_train.itertuples():
        filename_label[filename]=label

    decode_dic={}
    for key in encode_dic.keys():
        decode_dic[encode_dic[key]]=key
    #
    train_data=pd.read_csv("dataset/train_val_df_127_10.csv")
    test_data=pd.read_csv("dataset/test2_df_127_10.csv")
    #test_data_1=pd.read_csv("dataset/test1_df_127_10.csv")
    #
    make_fold_txt()
    # 2.训练模型
    for fold in range(5):
        fold+=1
        model = fasttext.train_supervised('fastTextData_kfold/'+'train_'+str(fold)+'.txt', lr=0.5, epoch=10, wordNgrams=2)
        # 验证
        texts_test, labels_test = [], []
        with open('fastTextData_kfold/'+'val_'+str(fold)+'.txt', 'r', encoding='utf-8') as f:
            for line in f:
                *text, label = line.strip().split(' ')
                text = ' '.join(text)
                texts_test.append(text)
                labels_test.append(label)
        #
        label_encoder = preprocessing.LabelEncoder()
        labels_test = label_encoder.fit_transform(labels_test)
        predits = list(zip(*(model.predict(texts_test)[0])))[0]
        predits = label_encoder.transform(predits)
        score=metrics.accuracy_score(predits, labels_test)
        print('fold {} :val accuracy score : {}'.format(fold,score))
        save_model = 'model_save_dir/'
        if not os.path.exists(save_model):
            os.mkdir(save_model)
        save_model = os.path.join(save_model, 'fastext_'+str(fold)+'_' + str(int(score * 1000)) + '.bin')
        model.save_model(save_model)
        print(save_model)
    #
    model_names=os.listdir("model_save_dir/")
    model1 = fasttext.load_model(os.path.join("model_save_dir",model_names[0]))
    model2 = fasttext.load_model(os.path.join("model_save_dir",model_names[1]))
    model3 = fasttext.load_model(os.path.join("model_save_dir",model_names[2]))
    model4 = fasttext.load_model(os.path.join("model_save_dir",model_names[3]))
    model5 = fasttext.load_model(os.path.join("model_save_dir",model_names[4]))
    #
    #
    test_data = pd.read_csv("dataset/test2_df_127_10.csv")
    sub_sample = pd.read_csv('data/submit_example_test2.csv')
    sub_filenames = sub_sample['filename'].values
    pres = []
    test_paths = glob.glob('test2/*')
    test_paths = [sample.replace('\\', '/') for sample in test_paths]
    #
    index = 0
    for _, name, text in tqdm(test_data.itertuples()):
        # 结巴分词
        seg_text = jieba.cut(text)
        out_str = " ".join(seg_text)
        #
        vote1 = int(model1.predict(out_str)[0][0].split('_')[-1])
        vote2 = int(model2.predict(out_str)[0][0].split('_')[-1])
        vote3 = int(model3.predict(out_str)[0][0].split('_')[-1])
        vote4 = int(model4.predict(out_str)[0][0].split('_')[-1])
        vote5 = int(model5.predict(out_str)[0][0].split('_')[-1])
        tmp_list = {}
        for i in range(20):
            tmp_list[i] = 0
        #
        merges = [vote1, vote2, vote3, vote4, vote5]
        for k in merges:
            tmp_list[k] += 1
        # print(tmp)
        most = sorted(tmp_list.items(), key=lambda item: item[1])[-1][0]
        pres.append(most)
    #
    # 解码
    pres = [decode_dic[i] for i in pres]
    #
    sub_sample['label'] = pres
    sub_sample.to_csv('submit.csv', index=False)
    print(sub_sample)


