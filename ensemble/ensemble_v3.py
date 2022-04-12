"""
@Author: Du Yunhao
@Filename: ensemble_v3.py
@Contact: dyh_bupt@163.com
@Time: 2022/4/9 14:35
@Discription: model ensemble v2, 直接基于ranklist的融合
"""
import json
import torch
import numpy as np
from os.path import join
from collections import defaultdict
from utils import evaluate_recall_mrr

if __name__ == '__main__':
    root = '/data/dyh/checkpoints/AICity2022Track2'
    dir_save = '/data/dyh/results/AICity2022Track2/TEST'
    ranklist_1 = json.load(open(join(dir_save, 'Swin-B+CLIP-B_OMG2a_NLAug_IDLoss_fold4_Continue1+2.json'), 'r'))  # 高优先级
    ranklist_2 = json.load(open(join(dir_save, 'Swin-B+CLIP-B_OMG2a_NLAug_IDLoss_fold4.json'), 'r'))
    ranklist_1 = list(ranklist_1.items())
    ranklist_2 = list(ranklist_2.items())
    ranklist_res = dict()
    assert len(ranklist_1) == len(ranklist_2) == 184
    delta = 0.1
    for i in range(184):
        key1, value1 = ranklist_1[i]
        key2, value2 = ranklist_2[i]
        assert key1 == key2
        id2score = defaultdict(int)
        for j in range(184):
            id2score[value1[j]] += j
            id2score[value2[j]] += j + delta
        re_ranklist = sorted(list(value1), key=lambda x: id2score[x])
        ranklist_res[key1] = re_ranklist
    json.dump(ranklist_res, open(join(dir_save, 'test.json'), 'w'), indent=2)