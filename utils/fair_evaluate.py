
import sys
import pdb
import math
import torch
import numpy as np
import multiprocessing as mp
from .metric import *

def compute_precision_at_k(topk_items, test_u2i, k):
    precisions = []
    for user_id, rec_items in enumerate(topk_items):
        rec_k = rec_items[:k]
        true_items = set(test_u2i.get(user_id, []))
        if not true_items:
            continue
        hit_count = sum(1 for item in rec_k if item in true_items)
        precisions.append(hit_count / k)
    return np.mean(precisions) if precisions else 0.0

def ranking_evaluate(user_emb, item_emb, n_users, n_items, train_u2i, test_u2i, sens=None, indicators='[\'ndcg\', '
                                                                                                      '\'recall\','
                                                                                                      '\'mrr\','
                                                                                                      '\'precision\']',
                     topks='[10, 20, 30]', num_workers=4):
    indicators = eval(indicators)
    topks = eval(topks)
    scores = np.matmul(user_emb, item_emb.T)

    # usuniecie precision z eval_accelerate
    # indicators_no_prec = [ind for ind in indicators if ind != 'precision']

    perf_info, topk_items = eval_accelerate(scores, n_users, train_u2i, test_u2i, indicators, topks, num_workers)
    # perf_info, topk_items = eval_accelerate(scores, n_users, train_u2i, test_u2i, indicators_no_prec, topks, num_workers)
    perf_info = np.mean(perf_info, axis=0)

    res = {}
    k = 0
    for ind in indicators:
        for topk in topks:
            res[ind + '@' + str(topk)] = perf_info[k]
            k = k + 1

    # ręczne obliczenie precision@k
    # if 'precision' in indicators:
    #     for topk in topks:
    #         res['precision@' + str(topk)] = compute_precision_at_k(topk_items, test_u2i, topk)

    if 'precision' in indicators and 'recall' in indicators:
        for topk in topks:
            precision = res['precision@' + str(topk)]
            recall = res['recall@' + str(topk)]
            res['f1@' + str(topk)] = 2 * precision * recall / (precision + recall)

    if sens is not None:
        for topk in topks:
            res['js_dp@' + str(topk)], res['js_eo@' + str(topk)] = js_topk(topk_items, sens, test_u2i, n_users, n_items, topk)
    return res


def ranking_age_evaluate(user_emb, item_emb, n_users, n_items, train_u2i, test_u2i, sens=None, sens_age=None, sens_age6=None, indicators='[\'ndcg\', '
                                                                                                      '\'recall\', \'mrr\', \'precision\']',
                     topks='[10, 20, 30]', num_workers=4):
    indicators = eval(indicators)
    topks = eval(topks)
    scores = np.matmul(user_emb, item_emb.T)

    # indicators_no_prec = [ind for ind in indicators if ind != 'precision']

    perf_info, topk_items = eval_accelerate(scores, n_users, train_u2i, test_u2i, indicators, topks, num_workers)
    perf_info = np.mean(perf_info, axis=0)

    res = {}
    k = 0
    for ind in indicators:
        for topk in topks:
            res[ind + '@' + str(topk)] = perf_info[k]
            k = k + 1

    # if 'precision' in indicators:
    #     for topk in topks:
    #         res['precision@' + str(topk)] = compute_precision_at_k(topk_items, test_u2i, topk)

    if 'precision' in indicators and 'recall' in indicators:
        for topk in topks:
            precision = res['precision@' + str(topk)]
            recall = res['recall@' + str(topk)]
            res['f1@' + str(topk)] = 2 * precision * recall / (precision + recall)


    if sens is not None:
        for topk in topks:
            res['js_dp@' + str(topk)], res['js_eo@' + str(topk)] = (
                js_topk(topk_items, sens,test_u2i,n_users, n_items, topk))
    if sens_age is not None:
        for topk in topks:
            res['age_dp@' + str(topk)], res['age_eo@' + str(topk)] = (
                js_topk(topk_items, sens_age,test_u2i,n_users, n_items, topk))

    if sens_age6 is not None:
        for topk in topks:
             res['age6_dp@' + str(topk)], res['age6_eo@' + str(topk)] = (
                js_topk_multi(topk_items, sens_age6,test_u2i,n_users, n_items, topk))

    return res

def eval_accelerate(scores, n_users, train_u2i, test_u2i, indicators, topks, num_workers):
    test_user_set = list(test_u2i.keys())
    perf_info = np.zeros(shape=(len(test_user_set), len(topks) * len(indicators)), dtype=np.float32)
    topk_items = np.zeros(shape=(n_users, max(topks)), dtype=np.int32)

    test_parameters = zip(test_user_set, )

    if sys.platform.startswith('win32'):
        res = []
        _init_global(scores, train_u2i, test_u2i, indicators, topks)
        for param in test_parameters:
            one_result = test_one_perf(param)
            res.append(one_result)
    else:
        with mp.Pool(processes=num_workers, initializer=_init_global,
                     initargs=(scores, train_u2i, test_u2i, indicators, topks,)) as pool:
            res = pool.map(test_one_perf, test_parameters)

    for i, one in enumerate(res):
        perf_info[i] = one[0]
        topk_items[one[1][0]] = one[1][1:] #one[1][0]=user_id;one[1][1:]=top_30

    return perf_info, topk_items


def _init_global(_scores, _train_u2i, _test_u2i, _indicators, _topks):
    global scores, train_u2i, test_u2i, indicators, topks

    scores = _scores
    train_u2i = _train_u2i
    test_u2i = _test_u2i
    indicators = _indicators
    topks = _topks


def test_one_perf(x):
    u_id = x[0]
    score = np.copy(scores[u_id])
    uid_train_pos_items = list(train_u2i[u_id])
    uid_test_pos_items = list(test_u2i[u_id])
    score[uid_train_pos_items] = -np.inf
    score_indices = largest_indices(score, topks)
    res1 = get_perf(score_indices, uid_test_pos_items, topks)
    res2 = get_topks_items(u_id, score_indices)
    return (res1, res2)


def largest_indices(score, topks):
    max_topk = max(topks)
    indices = np.argpartition(score, -max_topk)[-max_topk:]
    indices = indices[np.argsort(-score[indices])]
    return indices


def get_perf(rank, uid_test_pos_items, topks):
    topk_eval = np.zeros(len(indicators) * len(topks), dtype=np.float32)
    k = 0
    for ind in indicators:
        for topk in topks:
            topk_eval[k] = eval(ind)(rank[:topk], uid_test_pos_items)
            k = k + 1
    return topk_eval


def get_topks_items(uid, rank):
    max_topk = max(topks)
    topk_items = rank[:max_topk]
    return np.hstack([np.array(uid, dtype=np.int32), topk_items])
