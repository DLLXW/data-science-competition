import pandas as pd
import jieba
import os
import glob
from tqdm import tqdm
import re
import xlrd
import json
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
def get_train_data(NROWS,topk):
    #
    print("读取训练数据－－－－－－－－－－－－－")
    # 获取每个文件的内容，和文件标题一起进行保存
    # save_data = 'fastTextData/'
    # if not os.path.exists(save_data):
    #     os.mkdir(save_data)
    train_val_df = {"text":[],"label":[]}
    #
    for per_path in tqdm(train_paths):
        # 结巴分词
        text_title = per_path.split('/')[-1]
        text_title = clean_data(text_title)
        flag = filename_label[per_path]
        text_content=[]
        if per_path not in train_error_clean:  # html文件直接跳过
            if (per_path.split('.')[-1] == 'csv') or (per_path in train_normal_clean):
                tmp_csv = pd.read_csv(per_path).reset_index(drop=True)  # 这里必须reset_index,不然会有bug
                text_content = clean_data(''.join(tmp_csv[:NROWS].values.reshape(-1, ).astype(str).tolist()))
            else:
                bk = xlrd.open_workbook(per_path)
                sheet_cnt = len(bk.sheet_names())
                if sheet_cnt > 1:
                    sh = bk.sheet_by_name(bk.sheet_names()[1])
                else:
                    sh = bk.sheet_by_name(bk.sheet_names()[0])
                #
                nrows = sh.nrows
                ncols = sh.ncols
                tmp_xls = ""
                if nrows > NROWS:
                    nrows = NROWS
                for i in range(1, nrows):
                    tmp = [str(jj) for jj in sh.row_values(i)]
                    row_data = ''.join(tmp)
                    tmp_xls += row_data
                #
                text_content = clean_data(tmp_xls)
            #这里是提取关键词
            if len(text_content) > 0:  # 如果内容文件不为空
                text_content_count = list(jieba.cut(text_content))  # 分词结果
                counter = Counter(text_content_count)
                #前topk高频词
                text_content_count = "".join([ii[0] for ii in counter.most_common(topk)])
            else:
                text_content_count=""
        #
        if len(text_content) == 0:
            text_content=""
        text = text_title+text_content_count+text_content
        #text = text_content
        #
        if len(text)==0:
            text="工业"
        assert len(text) > 0
        #
        train_val_df["text"].append(text)
        train_val_df["label"].append(int(flag))
    #
    train_val_df=pd.DataFrame(train_val_df)
    train_val_df.to_csv("../dataset/train_val_df_127_"+str(NROWS)+".csv",index=False)

def get_test_data(NROWS,topk):
    print("读取测试数据－－－－－－－－－－－－－")
    test_data_df={"name":[],"text":[]}
    sub_sample=pd.read_csv("submit_example_test2.csv")
    sub_filenames=sub_sample['filename'].values
    for per_path in tqdm(sub_filenames):
        # 结巴分词
        text_title = per_path.split('/')[-1]
        text_title = clean_data(text_title)#
        #
        text_content = []
        if per_path not in test_error_clean:  #html文件直接跳过
            if (per_path.split('.')[-1] == 'csv') or (per_path in test_normal_clean):
                tmp_csv = pd.read_csv(per_path).reset_index(drop=True)  # 这里必须reset_index,不然会有bug
                text_content = clean_data(''.join(tmp_csv[:NROWS].values.reshape(-1, ).astype(str).tolist()))
            else:
                bk=xlrd.open_workbook(per_path)
                sheet_cnt= len(bk.sheet_names())
                if sheet_cnt>1:
                    sh = bk.sheet_by_name(bk.sheet_names()[1])
                else:
                    sh= bk.sheet_by_name(bk.sheet_names()[0])
                #
                nrows = sh.nrows
                ncols = sh.ncols
                tmp_xls = ""
                if nrows>NROWS:
                    nrows=NROWS
                for i in range(1, nrows):
                    tmp=[str(jj) for jj in sh.row_values(i)]
                    row_data = ''.join(tmp)
                    tmp_xls+=row_data
                #
                text_content = clean_data(tmp_xls)
            #这里是提取关键词
            if len(text_content) > 0:  # 如果内容文件不为空
                text_content_count = list(jieba.cut(text_content))  # 分词结果
                counter = Counter(text_content_count)
                #前topk高频词
                text_content_count = "".join([ii[0] for ii in counter.most_common(topk)])
            else:
                text_content_count=""
        #
        if len(text_content) == 0:
            text_content=""
        text = text_title+text_content_count+text_content
        #text = text_content
        if len(text)==0:
            text="工业"
        assert len(text) > 0
        #
        test_data_df["name"].append(per_path)
        test_data_df["text"].append(text)
    test_data_df = pd.DataFrame(test_data_df)
    test_data_df.to_csv("../dataset/test2_df_127_"+str(NROWS)+".csv", index=False)
#a榜数据用作辅助
def get_test_data_1(NROWS,topk):
    print("读取测试数据－－－－－－－－－－－－－")
    test_data_df={"name":[],"text":[]}
    sub_sample=pd.read_csv("submit_example_test1.csv")
    sub_filenames=sub_sample['filename'].values
    for per_path in tqdm(sub_filenames):
        # 结巴分词
        text_title = per_path.split('/')[-1]
        text_title = clean_data(text_title)#
        #
        text_content = []
        if per_path not in test_error_clean_1:  #html文件直接跳过
            if (per_path.split('.')[-1] == 'csv') or (per_path in test_normal_clean_1):
                tmp_csv = pd.read_csv(per_path).reset_index(drop=True)  # 这里必须reset_index,不然会有bug
                text_content = clean_data(''.join(tmp_csv[:NROWS].values.reshape(-1, ).astype(str).tolist()))
            else:
                bk=xlrd.open_workbook(per_path)
                sheet_cnt= len(bk.sheet_names())
                if sheet_cnt>1:
                    sh = bk.sheet_by_name(bk.sheet_names()[1])
                else:
                    sh= bk.sheet_by_name(bk.sheet_names()[0])
                #
                nrows = sh.nrows
                ncols = sh.ncols
                tmp_xls = ""
                if nrows>NROWS:
                    nrows=NROWS
                for i in range(1, nrows):
                    tmp=[str(jj) for jj in sh.row_values(i)]
                    row_data = ''.join(tmp)
                    tmp_xls+=row_data
                #
                text_content = clean_data(tmp_xls)
            #这里是提取关键词
            if len(text_content) > 0:  # 如果内容文件不为空
                text_content_count = list(jieba.cut(text_content))  # 分词结果
                counter = Counter(text_content_count)
                #前topk高频词
                text_content_count = "".join([ii[0] for ii in counter.most_common(topk)])
            else:
                text_content_count=""
        #
        if len(text_content) == 0:
            text_content=""
        text = text_title+text_content_count+text_content
        #text = text_content
        if len(text)==0:
            text="工业"
        assert len(text) > 0
        #
        test_data_df["name"].append(per_path)
        test_data_df["text"].append(text)
    test_data_df = pd.DataFrame(test_data_df)
    test_data_df.to_csv("./dataset/test1_df_127_"+str(NROWS)+".csv", index=False)
if __name__=="__main__":
    with open('../jsons/train_error_clean.json', 'r', encoding='UTF-8') as f:
        train_error_clean = json.load(f)
    with open('../jsons/train_normal_clean.json', 'r', encoding='UTF-8') as f:
        train_normal_clean = json.load(f)
    with open('../jsons/test_error_clean.json', 'r', encoding='UTF-8') as f:
        test_error_clean = json.load(f)
    with open('../jsons/test_normal_clean.json', 'r', encoding='UTF-8') as f:
        test_normal_clean = json.load(f)
    with open('../jsons/test_error_clean_1.json', 'r', encoding='UTF-8') as f:
        test_error_clean_1 = json.load(f)
    with open('../jsons/test_normal_clean_1.json', 'r', encoding='UTF-8') as f:
        test_normal_clean_1 = json.load(f)
    #
    #----------编码－－－－－－－－－－－－
    answer_train = pd.read_csv('answer_train.csv')
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
    #print(decode_dic)
    # #
    # test_paths_1 = glob.glob('test1/*')  # 获取测试集所以文件绝对路径
    # test_paths_1 = [sample.replace('\\', '/') for sample in test_paths_1]
    # get_test_data_1(10, 20)
    #
    test_paths = glob.glob('test2/*')  # 获取测试集所以文件绝对路径
    test_paths = [sample.replace('\\', '/') for sample in test_paths]
    get_test_data(10,20)
    # 获取每个文件的内容,和文件标题一起进行保存
    train_paths = glob.glob('train/*')  # 获取训练集所以文件绝对路径
    train_paths = [sample.replace('\\', '/') for sample in train_paths]
    get_train_data(10,20)

