"""
@Author: Du Yunhao
@Filename: ReRanking.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/20 10:25
@Discription: k-reciprocal re-ranking
解析：https://blog.csdn.net/u014453898/article/details/98790860
"""
import torch
import numpy as np

def re_ranking_similarity(features_probe, features_gallery, k1, k2, lambda_):
    assert torch.is_tensor(features_probe) and torch.is_tensor(features_gallery)
    '''初始化'''
    num_q, num_g = features_probe.size(0), features_gallery.size(0)
    num_all = num_q + num_g
    features = torch.cat([features_probe, features_gallery])
    similarity_cosine = features @ features.t()
    similarity_cosine = similarity_cosine.cpu().numpy()
    similarity_cosine = (similarity_cosine / similarity_cosine.max(axis=0)).T  # 相似度矩阵，每行归一化
    del features
    # similarity_cosine[:num_q, :num_q] = 0
    # similarity_cosine[:num_q, num_q:] = 0
    # similarity_cosine[num_q:, :num_q] = 0
    # similarity_cosine[num_q:, num_q:] = 0
    '''计算互近邻特征'''
    V = np.zeros_like(similarity_cosine).astype(np.float16)  # k-reciprocal features
    ranklist = similarity_cosine.argsort(axis=1)[:, ::-1]  # 排序列表
    for i in range(num_all):  # 遍历query
        q_k1_nearest_neighbors = ranklist[i, :k1+1]  # qi的k1最近邻，[k1+1,]，+1是因为会算进自己
        g_k1_nearest_neighbors = ranklist[q_k1_nearest_neighbors, :k1+1]  # qi的k1最近邻的k1最近邻，[k1+1,k1+1]
        k1_reciprocal = q_k1_nearest_neighbors[
            np.where(g_k1_nearest_neighbors == i)[0]
        ]  # qi的k1互近邻
        k1_reciprocal_expansion = k1_reciprocal  # qi的k1互近邻扩充
        for candidate in k1_reciprocal:  # 遍历互近邻gallery扩展互近邻集合
            candidate_q_k1d2_nearest_neighbors = ranklist[candidate, :k1//2 + 1]
            candidate_g_k1d2_nearest_neighbors = ranklist[candidate_q_k1d2_nearest_neighbors, :k1//2 + 1]
            candidate_k1d2_reciprocal = candidate_q_k1d2_nearest_neighbors[
                np.where(candidate_g_k1d2_nearest_neighbors == candidate)[0]
            ]  # gj的k1/2互近邻
            if len(np.intersect1d(candidate_k1d2_reciprocal, k1_reciprocal)) > 2 / 3 * len(candidate_k1d2_reciprocal):
                k1_reciprocal_expansion = np.append(k1_reciprocal_expansion, candidate_k1d2_reciprocal)  # 互近邻集合扩展
        k1_reciprocal_expansion = np.unique(k1_reciprocal_expansion)
        weight = np.exp(similarity_cosine[i, k1_reciprocal_expansion])
        V[i, k1_reciprocal_expansion] = weight / np.sum(weight)
    '''Local Query Expansion'''
    V_lqe = np.zeros_like(V, dtype=np.float16)
    for i in range(num_all):
        V_lqe[i, :] = np.mean(
            V[ranklist[i, :k2], :],
            axis=0
        )  # 用原排序列表的前k2个特征均值作为该query的互近邻特征
        V = V_lqe
    del V_lqe
    del ranklist
    '''计算Jaccard相似度'''
    similarity_cosine = similarity_cosine[:num_q]
    similarity_jaccard = np.zeros_like(similarity_cosine, dtype=np.float16)  # [num_q, num_q+num_g]
    index_inverse = [np.where(V[:, i] != 0)[0] for i in range(num_all)]  # qg[i] -> [qi, qj, qk,...]
    for i in range(num_q):
        min_ = np.zeros(shape=[1, num_all], dtype=np.float16)
        index_NonZero = np.where(V[i, :] != 0)[0]  # q[i] -> [qgi, qgj, qgk,...]
        index = [index_inverse[idx] for idx in index_NonZero]
        for j in range(len(index_NonZero)):
            min_[0, index[j]] = min_[0, index[j]] + \
                                np.minimum(
                                    V[i, index_NonZero[j]],
                                    V[index[j], index_NonZero[j]]
                                )
        similarity_jaccard[i] = min_ / (2 - min_)
    '''最终相似度'''
    similarity = lambda_ * similarity_cosine + (1 - lambda_) * similarity_jaccard
    similarity = similarity[:, num_q:]
    return similarity


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        distmat = 2 - feat.mm(feat.t())
        # distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
        #               torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        # distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))

    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist