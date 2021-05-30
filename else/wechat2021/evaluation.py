# coding: utf-8
import time
import traceback
from collections import defaultdict
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc


def compute_weighted_score(score_dict, weight_dict):
    '''基于多个行为的uAUC值，计算加权uAUC
    Input:
        scores_dict: 多个行为的uAUC值映射字典, dict
        weights_dict: 多个行为的权重映射字典, dict
    Output:
        score: 加权uAUC值, float
    '''
    score = 0.0
    weight_sum = 0.0
    for action in score_dict:
        weight = float(weight_dict[action])
        score += weight*score_dict[action]
        weight_sum += weight
    score /= float(weight_sum)
    score = round(score, 6)
    return score


def score(result_data, label_data, mode="初赛"):
    '''评测结果: 多个行为的加权uAUC分数
    Input:
        result_data: 提交的结果文件，二进制格式
        label_data: 对应的label文件，二进制格式
        mode: 比赛阶段，String. "初赛"/"复赛"
    Output:
        result: 评测结果，dict
    '''
    try:
        # 读取数据
        logger.info('Read data')
        result_df = pd.read_csv(result_data, sep=',')
        label_df = pd.read_csv(label_data, sep=',')
        if mode == "初赛":
            # 初赛只评估四个互动行为
            actions = ['read_comment', 'like', 'click_avatar', 'forward']
        else:
            # 复赛评估七个互动行为
            actions = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
        # 互动行为权重映射表
        weights_map = {
            "read_comment": 4.0,  # 是否查看评论
            "like": 3.0,  # 是否点赞
            "click_avatar": 2.0,  # 是否点击头像
            "forward": 1.0,  # 是否转发
            "favorite": 1.0,  # 是否收藏
            "comment": 1.0,  # 是否发表评论
            "follow": 1.0  # 是否关注
        }
        target_cols = ['userid', 'feedid'] + actions
        label_df = label_df[target_cols]
        # 规范检查
        logger.info('Check result file')
        if result_df.shape[0] != label_df.shape[0]:
            err_msg = "结果文件的行数（%i行）与测试集（%i行）不一致"%(result_df.shape[0], label_df.shape[0])
            res = {
                "ret": 1,
                "err_msg": err_msg,
            }
            logger.error(res)
            return res
        err_cols = []
        result_cols = set(result_df.columns)
        for col in target_cols:
            if col not in result_cols:
                err_cols.append(col)
        if len(err_cols) > 0:
            err_msg = "结果文件缺少字段/列：%s"%(', '.join(err_cols))
            res = {
                "ret": 2,
                "err_msg": err_msg,
            }
            logger.error(res)
            return res
        result_actions_map = {}
        label_actions_map = {}
        result_actions = []
        label_actions = []
        for action in actions:
            result_actions_map[action] = "result_"+action
            result_actions.append("result_"+action)
            label_actions_map[action] = "label_"+action
            label_actions.append("label_"+action)
        result_df = result_df.rename(columns=result_actions_map)
        label_df = label_df.rename(columns=label_actions_map)
        df = label_df.merge(result_df, on=['userid', 'feedid'])
        if len(df) != len(label_df):
            err_msg = "结果文件中userid-feedid与测试集不一致"
            res = {
                "ret": 3,
                "err_msg": err_msg,
            }
            logger.error(res)
            return res
        # 计算分数
        logger.info('Compute score')
        y_true = df[label_actions].astype(int).values
        y_pred = df[result_actions].astype(float).values.round(decimals=6)
        userid_list = df['userid'].astype(str).tolist()
        del df, result_df, label_df
        score = 0.0
        weights_sum = 0.0
        score_detail = {}
        for i, action in enumerate(actions):
            print(action)
            y_true_bev = y_true[:, i]
            y_pred_bev = y_pred[:, i]
            weight = weights_map[action]
            # user AUC
            uauc = uAUC(y_true_bev, y_pred_bev, userid_list)
            print(uauc)
            score_detail[action] = round(uauc, 6)
            score += weight*uauc
            weights_sum += weight
        score /= weights_sum
        score = round(score, 6)
        res = {
            "ret": 0,
            "data": {
                "score": score,
                "score_detail": score_detail
            }
        }
        logger.info(res)
    except Exception as e:
        traceback.print_exc()
        res = {
            "ret": 4,
            "err_msg": str(e)
        }    
        logger.error(res)
    return res

    
if __name__ == '__main__':
    t = time.time()
    label_data = open('data/evaluate/evaluate_all_13_generate_sample.csv', 'r')
    result_data = open('data/evaluate/submit_1619332123.csv', 'r')
    res = score(result_data, label_data, mode='初赛')
    print('Time cost: %.2f s'%(time.time()-t))
