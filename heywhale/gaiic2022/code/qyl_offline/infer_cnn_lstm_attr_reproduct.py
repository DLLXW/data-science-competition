
from net_CNN import CNN_Text
from net_BiLSTM import LSTM_Text
import torch
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import jieba
jieba.setLogLevel(jieba.logging.INFO)
def load_model(model_name,weight_path):
    print(weight_path)
    if model_name=='lstm':
        model=LSTM_Text(embed_num=embed_num,class_num=13)
    elif model_name=='cnn':
        model=CNN_Text(embed_num=embed_num,class_num=13)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict(text,feature):
    model=model_list_1[0]
    outputs=model(text,feature)
    pres=outputs.sigmoid().detach().cpu().numpy()[0]
    cnt=1
    for i in range(1,len(model_list_1)):
        model=model_list_1[i]
        outputs=model(text,feature)
        outputs=outputs.sigmoid().detach().cpu().numpy()[0]
        pres+=outputs
        cnt+=1
    for i in range(1,len(model_list_2)):
        model=model_list_2[i]
        outputs=model(text,feature)
        outputs=outputs.sigmoid().detach().cpu().numpy()[0]
        pres+=outputs
        cnt+=1
    pres/=cnt
    return pres


if __name__=="__main__":
    # 
    out_file='../../data/submission/submit_attr_qyl.txt'
    out_file_prob='../../tmp/data/submission/submit_attr_qyl_prob.txt'
    data_dir='../../data/contest_data/preliminary_testB.txt'
    #
    with open('../../data/word_to_idx_fine_coarse_attr_reproduct.json', 'r') as f:
        word_to_idx = json.load(f)
    embed_num=len(word_to_idx)+1
    print("embed_num:",embed_num)
    #
    device=torch.device('cuda')
    class_name=['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
    class_dict={'图文': ['符合','不符合'], 
                '版型': ['修身型', '宽松型', '标准型'], 
                '裤型': ['微喇裤', '小脚裤', '哈伦裤', '直筒裤', '阔腿裤', '铅笔裤', 'O型裤', '灯笼裤', '锥形裤', '喇叭裤', '工装裤', '背带裤', '紧身裤'],
                '袖长': ['长袖', '短袖', '七分袖', '五分袖', '无袖', '九分袖'], 
                '裙长': ['中长裙', '短裙', '超短裙', '中裙', '长裙'], 
                '领型': ['半高领', '高领', '翻领', 'POLO领', '立领', '连帽', '娃娃领', 'V领', '圆领', '西装领', '荷叶领', '围巾领', '棒球领', '方领', '可脱卸帽', '衬衫领', 'U型领', '堆堆领', '一字领', '亨利领', '斜领', '双层领'], 
                '裤门襟': ['系带', '松紧', '拉链'], 
                '鞋帮高度': ['低帮', '高帮', '中帮'], 
                '穿着方式': ['套头', '开衫'], 
                '衣长': ['常规款', '中长款', '长款', '短款', '超短款', '超长款'], 
                '闭合方式': ['系带', '套脚', '一脚蹬', '松紧带', '魔术贴', '搭扣', '套筒', '拉链'], 
                '裤长': ['九分裤', '长裤', '五分裤', '七分裤', '短裤'], 
                '类别': ['单肩包', '斜挎包', '双肩包', '手提包']
                }
    submit_sample={"img_name":"test000255","match":{"图文":0,"领型":1,"袖长":1,"穿着方式":0}}
    submit_sample_prob={"img_name":"test000255","match":{"图文":0,"领型":1,"袖长":1,"穿着方式":0}}
    submit=[]
    submit_prob=[]
    class_index=[]
    model_list_1=[]
    model_list_2=[]
    for i in range(5):
        model_list_1.append(load_model('cnn','../../data/model_data/ckpt_cnn_fine_coarse_attr_warmup/fold_'+str(i+1)+'_best.pth'))
    for i in range(4):
       model_list_2.append(load_model('lstm','../../data/model_data/ckpt_0416_lstm_fine_coarse_attr_warmup/fold_'+str(i+1)+'_best.pth'))
    #
    Threshold=0.5
    max_length=20
    with open(data_dir, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            img_name=item['img_name']
            title=item['title']
            query=item['query']
            feature=torch.tensor(item['feature'])
            feature=feature.unsqueeze(0)
            feature=feature.cuda().float()
            jieba_cut=jieba.cut(title)
            text=[]
            for w in jieba_cut:
                if w in word_to_idx:
                    text.append(word_to_idx[w])
                else:
                    text.append(0)
            if len(text)>max_length:
                text=text[:max_length]
            else:
                text=text+[0]*(max_length-len(text))
            text=torch.tensor(text)
            text=text.unsqueeze(0)
            text=text.type(torch.LongTensor).cuda()
            pre=predict(text,feature)
            tmp={}
            for que in query:
                inx=class_name.index(que)
                if pre[inx]>Threshold:
                    #print(pre[inx])
                    tmp[que]=1
                else:
                    tmp[que]=0
            submit_sample['img_name']=img_name
            submit_sample['match']=tmp
            #
            submit_sample_prob['img_name']=img_name
            submit_sample_prob['match']=[float(p) for p in pre][1:]
            #
            print(submit_sample)
            submit.append(json.dumps(submit_sample, ensure_ascii=False)+'\n')
            submit_prob.append(json.dumps(submit_sample_prob, ensure_ascii=False)+'\n')
    # 
    with open(out_file, 'w') as f:
        f.writelines(submit)
    with open(out_file_prob, 'w') as f:
        f.writelines(submit_prob)