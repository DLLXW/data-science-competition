#
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('max.columns', 100)
import os
from datetime import datetime

import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def filter_col_by_nan(df, ratio=0.05):
    cols = []
    for col in df.columns:
        if df[col].isna().mean() >= (1-ratio):
            cols.append(col)
    return cols
def tfidif_frt(train_df_scope,test_df_scope):
    stopwords = stopwords = [line.strip() for line in open('D:/tianma/stopwords-master/cn_stopwords.txt',encoding='UTF-8').readlines()]
    stopwords+=['、', '；', '，', '）','（']
    #
    str_label_0=''
    str_label_1=''
    for index,name,opscope,label in train_df_scope.itertuples():
        # 结巴分词
        seg_text = jieba.cut(opscope.replace("\t", " ").replace("\n", " "))
        outline = " ".join(seg_text)
        out_str=""
        for per in outline.split():
            if per not in stopwords:
                out_str += per
                out_str+=" "
        if label==0:
            str_label_0+=out_str
        else:
            str_label_1+=out_str
    corpus=[str_label_0,str_label_1]
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语总共7175个词语
    weight=tfidf.toarray()#将(2, 7175)tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     #
    #     for j in range(len(word)):
    #         print(word[j],weight[i][j])
    #下面将会根据tfidi算出来的权重将经营范围的文本特征转换为数值(利用weight[1,:]也即各个词语在第二类(违法类中所占据的权重之和))
    illegal_word_weights={}
    for i in range(len(word)):
        illegal_word_weights[word[i]]=weight[1][i]
    tfidi_opscope=[]
    for index,name,opscope in base_info[['id', 'opscope']].itertuples():
        #
        seg_text = jieba.cut(opscope.replace("\t", " ").replace("\n", " "))
        outline = " ".join(seg_text)
        tfidi_frt=0
        for per in outline.split():
            if per in illegal_word_weights:
                tfidi_frt+=illegal_word_weights[per]
        tfidi_opscope.append(tfidi_frt)
    return tfidi_opscope#返回的是一个列表

#
def get_base_info_frts(base_info):
    base_info = base_info.drop(filter_col_by_nan(base_info, 0.01), axis=1)
    #----------------交叉特征 add by qyl-----------------
    #行业类别_行业细类-348
    #base_info['industryphy_industryco']=base_info['industryphy'].astype(str) + '_' + base_info['industryco'].astype(str)
    #企业类型_企业类型小类 37
    base_info['enttype_enttypeitem']=base_info['enttype'].astype(str) + '_' + base_info['enttypeitem'].astype(str)
    #企业(机构)类型--企业类型细类 53
    base_info['enttypegb_enttypeminu']=base_info['enttypegb'].astype(str) + '_' + base_info['enttypeminu'].astype(str)
    #行业类别_企业类型 99
    base_info['industryphy_enttype']=base_info['industryphy'].astype(str) + '_' + base_info['enttype'].astype(str)
    #1072
    base_info['enttype_enttypeitem_enttypegb_enttypeminu']=base_info['enttype_enttypeitem'].astype(str) + '_' + base_info['enttypegb_enttypeminu'].astype(str)
    #
    #行政区划代码/行政标识的前6位
    base_info['district_FLAG1'] = (base_info['orgid'].fillna('').apply(lambda x: str(x)[:6]) == \
        base_info['oplocdistrict'].fillna('').apply(lambda x: str(x)[:6])).astype(int)
    base_info['district_FLAG2'] = (base_info['orgid'].fillna('').apply(lambda x: str(x)[:6]) == \
        base_info['jobid'].fillna('').apply(lambda x: str(x)[:6])).astype(int)
    base_info['district_FLAG3'] = (base_info['oplocdistrict'].fillna('').apply(lambda x: str(x)[:6]) == \
        base_info['jobid'].fillna('').apply(lambda x: str(x)[:6])).astype(int)

    base_info['person_SUM'] = base_info[['empnum', 'parnum', 'exenum']].sum(1)
    base_info['person_NULL_SUM'] = base_info[['empnum', 'parnum', 'exenum']].isnull().astype(int).sum(1)
    #------add by qyl--------------
    #经营地址编码虽然不同，但是长度变化却很小.总共长度只有21种，并且都可以被16整除，前32位不同的只有412种
    #对于id特征亦如此，长度都为48，所以可以先对id按16位进行切分
    #pd.Series([len(i) for i in base_info['dom'].values]).unique()/16
    base_info['id_prefix']=[per[:16] for per in base_info['id'].values]
    base_info['dom_prefix']=[per[:32] for per in base_info['dom'].values]
    #-----------------add by qyl----------------
    # base_info['regcap_DIVDE_empnum'] = base_info['regcap'] / base_info['empnum']
    # base_info['regcap_DIVDE_exenum'] = base_info['regcap'] / base_info['exenum']

    # base_info['reccap_DIVDE_empnum'] = base_info['reccap'] / base_info['empnum']
    # base_info['regcap_DIVDE_exenum'] = base_info['regcap'] / base_info['exenum']

    # base_info['congro_DIVDE_empnum'] = base_info['congro'] / base_info['empnum']
    # base_info['regcap_DIVDE_exenum'] = base_info['regcap'] / base_info['exenum']

    base_info['opfrom'] = pd.to_datetime(base_info['opfrom'])
    base_info['opto'] = pd.to_datetime(base_info['opto'])
    base_info['opfrom_TONOW'] = (datetime.now() - base_info['opfrom']).dt.days
    base_info['opfrom_TIME'] = (base_info['opto'] - base_info['opfrom']).dt.days

    base_info['opscope_COUNT'] = base_info['opscope'].apply(lambda x: len(x.replace("\t", "，").replace("\n", "，").split('、')))

    cat_col = ['oplocdistrict', 'industryphy', 'industryco', 'enttype',
               'enttypeitem', 'enttypeminu', 'enttypegb',
              'oploc', 'opform',
              'dom_prefix','id_prefix',
              #'industryphy_industryco',
               'enttype_enttypeitem','enttypegb_enttypeminu',
              'industryphy_enttype',
               'enttype_enttypeitem_enttypegb_enttypeminu'
              ]

    for col in cat_col:
        #将每一个类别都映射为其出现的次数，将出现次数<10的类别用-1来代替
        base_info[col + '_COUNT'] = base_info[col].map(base_info[col].value_counts())
        col_idx = base_info[col].value_counts()
        for idx in col_idx[col_idx < 10].index:
            base_info[col] = base_info[col].replace(idx, -1)

    # base_info['opscope'] = base_info['opscope'].apply(lambda x: x.replace("\t", " ").replace("\n", " ").replace("，", " "))
    # clf_tfidf = TfidfVectorizer(max_features=200)
    # tfidf=clf_tfidf.fit_transform(base_info['opscope'])
    # tfidf = pd.DataFrame(tfidf.toarray())
    # tfidf.columns = ['opscope_' + str(x) for x in range(200)]
    # base_info = pd.concat([base_info, tfidf], axis=1)

    base_info = base_info.drop(['opscope','opfrom', 'opto','dom'], axis=1)

    for col in ['industryphy','opform', 'oploc','id_prefix','dom_prefix',
                #'industryphy_industryco',
                'enttype_enttypeitem','enttypegb_enttypeminu',
                'industryphy_enttype',
                'enttype_enttypeitem_enttypegb_enttypeminu'
               ]:
        base_info[col] = pd.factorize(base_info[col])[0]

    print("get base_info frts finished ............")
    return base_info
#

def get_others_frts():
    # 剔除纯空列
    annual_report_info = pd.read_csv(PATH + 'annual_report_info.csv')
    annual_report_info.drop(filter_col_by_nan(annual_report_info, 0.01), axis=1,inplace=True)

    other_info = pd.read_csv(PATH + 'other_info.csv')
    other_info_df = other_info[~other_info['id'].duplicated()]  # 去重复列
    other_info_df['other_SUM'] = other_info_df[['legal_judgment_num', 'brand_num', 'patent_num']].sum(1)  # 每一行的和
    other_info_df['other_NULL_SUM'] = other_info_df[['legal_judgment_num', 'brand_num', 'patent_num']].isnull().astype(
        int).sum(1)  # 每一行多少空值

    # ---------news_info--------------
    news_info = pd.read_csv(PATH + 'news_info.csv')
    news_info['positive_negtive'] = news_info['positive_negtive'].fillna("中立")
    #
    dic_map = {"消极": 0, "中立": 1, "积极": 2}
    #
    news_info['positive_negtive'] = news_info['positive_negtive'].map(dic_map)
    # news_info = news_info.groupby('id',sort=False).agg('mean')
    news_info['public_date'] = news_info['public_date'].apply(lambda x: x if '-' in str(x) else np.nan)  # 一些异常值需要处理掉
    news_info['public_date'] = pd.to_datetime(news_info['public_date'])
    news_info['public_date'] = (datetime.now() - news_info['public_date']).dt.days

    news_info_df = news_info.groupby('id').agg({'public_date': ['count', 'max', 'min', 'mean'],
                                                'positive_negtive': ['max', 'min', 'mean']}).reset_index()
    news_info_df.columns = ['id', 'public_date_COUNT', 'public_MAX', 'public_MIN', 'public_MEAN'
        , 'p_n_MAX', 'p_n_public_MIN', 'p_n_public_MEAN']
    news_info_df2 = pd.pivot_table(news_info, index='id', columns='positive_negtive', aggfunc='count').reset_index()
    news_info_df2.columns = ['id', 'news_COUNT1', 'news_COUNT2', 'news_COUNT3']
    news_info_df = pd.merge(news_info_df, news_info_df2)

    # ---------tax_info--------------
    tax_info = pd.read_csv(PATH + 'tax_info.csv')
    tax_info_df = tax_info.groupby('id').agg({
        'TAX_CATEGORIES': ['count', 'nunique'],
        'TAX_ITEMS': ['nunique'],
        'TAXATION_BASIS': ['nunique'],
        'TAX_AMOUNT': ['max', 'min', 'mean'],
        'TAX_RATE': ['max', 'min', 'mean']
    })
    tax_info_df.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper()
                                    for e in tax_info_df.columns.tolist()])
    tax_info_df = tax_info_df.reset_index()

    # ---------change_info--------------
    # change_info
    # [id:企业唯一标识, bgxmdm:变更信息代码, bgq:变更前, bgh:变更后, bgrq:变更日期]
    change_info = pd.read_csv(PATH + 'change_info.csv')
    change_info['bgrq'] = (change_info['bgrq'] / 10000000000).astype(int)

    change_info_df = change_info.groupby('id').agg({
        'bgxmdm': ['count', 'nunique'],
        'bgq': ['nunique'],
        'bgh': ['nunique'],
        'bgrq': ['nunique'],
    })
    change_info_df.columns = pd.Index(['changeinfo_' + e[0] + "_" + e[1].upper()
                                       for e in change_info_df.columns.tolist()])
    change_info_df = change_info_df.reset_index()
    change_info_df['bgq_bgh'] = change_info_df['changeinfo_bgq_NUNIQUE'] + change_info_df['changeinfo_bgh_NUNIQUE']
    #
    annual_report_info = pd.read_csv(PATH + 'annual_report_info.csv')
    annual_report_info_df = annual_report_info.groupby('id').agg({
        'ANCHEYEAR': ['max'],
        'STATE': ['max'],
        'FUNDAM': ['max'],
        'EMPNUM': ['max'],
        'UNEEMPLNUM': ['max', 'sum'],
        'WEBSITSIGN': ['max'],
        'PUBSTATE': ['max']
    })
    annual_report_info_df.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper()
                                              for e in annual_report_info_df.columns.tolist()])
    annual_report_info_df = annual_report_info_df.reset_index()
    return other_info_df, news_info_df, tax_info_df, change_info_df, annual_report_info_df

def merge_dataframe():
    train_data = pd.merge(base_info, entprise_info, on='id')
    train_data = pd.merge(train_data, other_info_df, on='id', how='left')

    train_data = pd.merge(train_data, news_info_df, on='id', how='left')
    train_data = pd.merge(train_data, tax_info_df, on='id', how='left')
    train_data = pd.merge(train_data, annual_report_info_df, on='id', how='left')
    train_data = pd.merge(train_data, change_info_df, on='id', how='left')

    entprise_evaluate = pd.read_csv('../data/' + 'entprise_evaluate.csv')
    entprise_evaluate = entprise_evaluate[['id']]
    test_data = pd.merge(base_info, entprise_evaluate, on='id')
    test_data = pd.merge(test_data, other_info_df, on='id', how='left')
    test_data = pd.merge(test_data, news_info_df, on='id', how='left')
    test_data = pd.merge(test_data, tax_info_df, on='id', how='left')
    test_data = pd.merge(test_data, annual_report_info_df, on='id', how='left')
    test_data = pd.merge(test_data, change_info_df, on='id', how='left')
    return train_data,test_data
def get_target_encode(train_data,test_data,cols):
    for col in cols:
        train_data[col+"_TargetRank"]=train_data[col].map(train_data.groupby([col])['label'].mean().rank())
        test_data[col+"_TargetRank"]=test_data[col].map(train_data.groupby([col])['label'].mean().rank())
    return train_data,test_data


if __name__=="__main__":
    PATH = '../data/train/'
    usr_data_dir = "../usr_data/"
    base_info = pd.read_csv(PATH + 'base_info.csv')
    annual_report_info = pd.read_csv(PATH + 'annual_report_info.csv')
    tax_info = pd.read_csv(PATH + 'tax_info.csv')
    change_info = pd.read_csv(PATH + 'change_info.csv')
    news_info = pd.read_csv(PATH + 'news_info.csv')
    other_info = pd.read_csv(PATH + 'other_info.csv')
    entprise_info = pd.read_csv(PATH + 'entprise_info.csv')
    entprise_evaluate = pd.read_csv('../data/' + 'entprise_evaluate.csv')
    #------------------对opscope提取tfidif特征------------------
    print('对opscope提取tfidif特征完毕..........')
    train_df_scope=base_info.merge(entprise_info)[['id', 'opscope', 'label']]
    test_df_scope=base_info[base_info['id'].isin(entprise_evaluate['id'].unique().tolist())]
    test_df_scope=test_df_scope.reset_index(drop=True)[['id','opscope']]
    base_info['tfidif_opscope']=tfidif_frt(train_df_scope,test_df_scope)
    print('对opscope提取tfidif特征完毕..........')
    #------------------
    base_info=get_base_info_frts(base_info)
    #
    other_info_df, news_info_df, tax_info_df, change_info_df, annual_report_info_df=get_others_frts()
    train_data,test_data=merge_dataframe()
    #target_编码
    target_cols=['enttype_enttypeitem','enttypegb_enttypeminu']
    train_data,test_data=get_target_encode(train_data,test_data,target_cols)
    #将特征暂时存储于usr_data_dir下
    if not os.path.exists(usr_data_dir):
        os.makedirs(usr_data_dir)
    print(train_data.shape,test_data.shape)
    train_data.to_csv(usr_data_dir+"train_data.csv",index=False)
    test_data.to_csv(usr_data_dir+"test_data.csv", index=False)
    #特征筛选