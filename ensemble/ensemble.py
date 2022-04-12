"""
@Author: Du Yunhao
@Filename: ensemble.py
@Contact: dyh_bupt@163.com
@Time: 2022/4/3 11:18
@Discription: model ensemble
"""
import json
import torch
import numpy as np
from os.path import join
from utils import evaluate_recall_mrr

def softmax(x):
    x = np.array(x)
    tau = 10
    return np.exp(tau * x) / np.sum(np.exp(tau * x))

if __name__ == '__main__':
    dim = 1024
    root = '/data/dyh/checkpoints/AICity2022Track2'
    methods = [
        'Res50IBN+CLIP-B_OMG2a_fold4',
        'Res50IBN+CLIP-B_OMG2a_NLAug_fold4_',
        # 'Res50IBN+CLIP-B_OMG2a_Size432_fold4',
    ]
    scores = [
        0.316,
        0.401,
        # 0.325
    ]

    methods = [
        'Swin-B+CLIP-B_OMG2a_NLAug_IDLoss_Continue_fold4',
        'Swin-B+CLIP-B_OMG2a_NLAug_IDLoss_Continue2_fold4',
        'Swin-B+CLIP-B_OMG2a_NLAug_IDLoss_fold4',
    ]
    scores = [
        1,
        1,
        1.5,
    ]

    # weights = softmax(scores)
    weights = scores
    length = len(weights)
    similaritys = []
    for method in methods:
        dir_ = join(root, method)
        ft = torch.tensor(
            np.load(join(
                dir_,
                # 'FeaturesText_val.npy'
                'Test_FeaturesText.npy'
            ))
        )
        fi = torch.tensor(
            np.load(join(
                dir_,
                # 'FeaturesCrop_val.npy'
                'Test_FeaturesImg.npy'
            ))
        )
        num_t = ft.size(1) // dim
        num_i = fi.size(1) // dim
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
        similaritys.append(similarity)

    similarity = torch.sum(torch.stack([weights[i] * similaritys[i] for i in range(length)]), dim=0)

    # gt = torch.arange(similarity.shape[0])
    # r1, r5, r10, mrr = evaluate_recall_mrr(similarity, gt)
    # print('{}th epoch: R@1 {} | R@5 {} | R@10 {} | MRR {}'.format('*', r1, r5, r10, mrr))

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