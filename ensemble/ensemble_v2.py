"""
@Author: Du Yunhao
@Filename: ensemble_v2.py
@Contact: dyh_bupt@163.com
@Time: 2022/4/7 9:58
@Discription: model ensemble v2, 保留强模型Topk
"""
import json
import torch
import numpy as np
from os.path import join
from utils import evaluate_recall_mrr

def get_similarity(dir_, test=False):
    if test:
        ft = torch.tensor(np.load(join(dir_, 'Test_FeaturesText.npy')))
        fi = torch.tensor(np.load(join(dir_, 'Test_FeaturesImg.npy')))
    else:
        ft = torch.tensor(np.load(join(dir_, 'FeaturesText_val.npy')))
        fi = torch.tensor(np.load(join(dir_, 'FeaturesCrop_val.npy')))
    num_t = ft.size(1) // dim
    num_i = fi.size(1) // dim
    print(num_t, num_i)
    ft = torch.mean(
        torch.stack([
            ft[:, i * dim: (i + 1) * dim] for i in range(num_t)
        ]),
        dim=0
    )
    fi = torch.mean(
        torch.stack([
            fi[:, i * dim: (i + 1) * dim] for i in range(num_i)
        ]),
        dim=0
    )
    similarity = ft @ fi.t()
    return similarity

def evaluate(similairty):
    gt = torch.arange(similarity.shape[0])
    r1, r5, r10, mrr = evaluate_recall_mrr(similarity, gt)
    print('{}th epoch: R@1 {} | R@5 {} | R@10 {} | MRR {}'.format('*', r1, r5, r10, mrr))

def save_test(similarity):
    dir_save = '/data/dyh/results/AICity2022Track2/TEST'
    similarity = similarity.numpy()
    uuidText = np.array(json.load(open(join(dir_save, 'Test_uuidText.json'), 'r')))
    uuidImg = np.array(json.load(open(join(dir_save, 'Test_uuidImg.json'), 'r')))
    results = dict()
    for i, sim in enumerate(similarity):
        idx = sim.argsort()[::-1]
        print(sim[idx])
        results[uuidText[i]] = uuidImg[idx].tolist()
    json.dump(results, open(join(dir_save, 'Test_results.json'), 'w'), indent=2)

if __name__ == '__main__':
    dim = 1024
    root = '/data/dyh/checkpoints/AICity2022Track2'
    method1 = 'Res50IBN+CLIP-B_OMG2a_NLAug_fold4'
    method2 = 'ZBY@1_0.3799_0.2865'

    test = True

    similarity1 = get_similarity(join(root, method1), test)
    similarity2 = get_similarity(join(root, method2), test)
    print(similarity1.size())

    k = 1
    index = similarity1.topk(k, dim=1)
    for sim, idx in zip(similarity1, index[1]):
        for idx_ in idx:
            sim[idx_] *= 1e5

    similarity = (similarity1 + similarity2) / 2

    if test:
        save_test(similarity)
    else:
        evaluate(similarity)




