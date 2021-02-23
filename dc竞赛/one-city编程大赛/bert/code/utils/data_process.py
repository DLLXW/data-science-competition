import pandas as pd
import jieba
import os
import glob
from tqdm import tqdm
import re
from collections import Counter
#这一步将可以正常读取的文件筛选出来
def dis_error_normal(paths):
    error=[]
    normal=[]
    for i in tqdm(range(len(paths))):
        name=paths[i]
        try:
            tmp=pd.read_csv(name)
            normal.append(name)
        except:
            error.append(name)
            continue
    return error,normal
#只筛选中文
def clean_data(data_in):
    # coding=utf-8
    if data_in=="":
        return ""
    pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
    data_out = pchinese.findall(data_in)
    if data_out:
        data_out = ''.join(data_out)
    if data_out == []:
        return ""
    return data_out
#
def get_train_data():
    #
    print("读取训练数据－－－－－－－－－－－－－")
    # 获取每个文件的内容，和文件标题一起进行保存
    save_data = 'fastTextData/'
    if not os.path.exists(save_data):
        os.mkdir(save_data)
    train_val_df = {"text":[],"label":[]}
    #
    for per_path in tqdm(train_paths):
        # 结巴分词
        text_title = per_path.split('/')[-1]
        text_title = clean_data(text_title)
        flag = filename_label[per_path]
        if use_only_head:
            text=text_title
        else:
            text_content = []
            if per_path in train_normal:  #
                tmp_csv = pd.read_csv(per_path).reset_index(drop=True)  # 这里必须reset_index,不然会有bug
                text_content = clean_data(''.join(tmp_csv[:100].values.reshape(-1, ).astype(str).tolist()))
                if len(text_content) > 0:  # 如果内容文件不为空
                    text_content = list(jieba.cut(text_content))  # 分词结果
                    counter = Counter(text_content)
                    # 前5高频词
                    text_content = [ii[0] for ii in counter.most_common(20)]
                else:
                    text_content = []
            #
            text_content="".join(text_content)#将列表转化为字符串
            text = text_title + text_content
        #
        if len(text)==0:
            text="工业"
        assert len(text) > 0
        #
        train_val_df["text"].append(text)
        train_val_df["label"].append(int(flag))
    #
    train_val_df=pd.DataFrame(train_val_df)
    train_val_df.to_csv("../../usr_data/train_val_df.csv",index=False)

def get_test_data():
    print("读取测试数据－－－－－－－－－－－－－")
    test_data_df={"name":[],"text":[]}
    sub_sample=pd.read_csv("../../usr_data/submit_example_test1.csv")
    sub_filenames=sub_sample['filename'].values
    for per_path in tqdm(sub_filenames):
        # 结巴分词
        text_title = per_path.split('/')[-1]
        text_title = clean_data(text_title)
        if use_only_head:
            text=text_title
        else:
            text_content = []
            if per_path in test_normal:  #
                tmp_csv = pd.read_csv(per_path).reset_index(drop=True)  # 这里必须reset_index,不然会有bug
                text_content = clean_data(''.join(tmp_csv[:100].values.reshape(-1, ).astype(str).tolist()))
                if len(text_content) > 0:  # 如果内容文件不为空
                    text_content = list(jieba.cut(text_content))  # 分词结果
                    counter = Counter(text_content)
                    # 前5高频词
                    text_content = [ii[0] for ii in counter.most_common(20)]
                else:
                    text_content = []
            #
            text_content="".join(text_content)#将列表转化为字符串
            text = text_title + text_content
        if len(text)==0:
            text="工业"
        assert len(text) > 0
        #
        test_data_df["name"].append(per_path)
        test_data_df["text"].append(text)
    test_data_df = pd.DataFrame(test_data_df)
    test_data_df.to_csv("../../usr_data/test_data_df_A.csv", index=False)

if __name__=="__main__":
    #----------编码－－－－－－－－－－－－
    answer_train = pd.read_csv('../../usr_data/answer_train.csv')
    encode_dic = {}
    cate = answer_train.label.unique()
    for i in range(len(cate)):
        encode_dic[cate[i]] = i
    answer_train['label'] = answer_train['label'].map(encode_dic)
    filename_label = {}
    for index, filename, label in answer_train.itertuples():
        filename_label[filename] = label
    #---------解码---------
    decode_dic = {}
    for key in encode_dic.keys():
        decode_dic[encode_dic[key]] = key
    print(decode_dic)
    #
    # 获取每个文件的内容,和文件标题一起进行保存
    use_only_head=True
    train_paths = glob.glob('train/*')  # 获取训练集所以文件绝对路径
    train_paths = [sample.replace('\\', '/') for sample in train_paths]
    test_paths = glob.glob('test1/*')  # 获取测试集所以文件绝对路径
    test_paths = [sample.replace('\\', '/') for sample in test_paths]
    train_error,train_normal=dis_error_normal(train_paths)
    test_error,test_normal=dis_error_normal(test_paths)
    #
    get_train_data()
    get_test_data()