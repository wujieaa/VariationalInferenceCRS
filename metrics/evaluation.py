from tqdm import tqdm

import numpy as np
import pandas as pd
from data_model.data_const import *


def recallk(true_num, hits, **unused):
    hits = sum(hits)
    return float(hits)/true_num


def precisionk(hits, **unused):
    hits_num = sum(hits)
    return float(hits_num)/len(hits)


def average_precisionk(hits, **unused):
    precisions = np.cumsum(hits, dtype=np.float32)/range(1, len(hits)+1)
    return np.mean(precisions)


def r_precision(true_num,hits, **unused):
    hits = hits[:true_num]
    hits = sum(hits)
    return float(hits)/true_num


def _dcg_support(size):
    arr = np.arange(1, size+1)+1
    return 1./np.log2(arr)


def ndcg(true_num, hits):
    idcg = np.sum(_dcg_support(true_num))
    dcg_base = _dcg_support(len(hits))
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg/idcg


def click(hits, **unused):
    first_hit = next((i for i, x in enumerate(hits) if x), None)
    if first_hit is None:
        return 5
    else:
        return first_hit/10


def evaluate(hits, pos_num,metric_names, atK, analytical=False):

    global_metrics = {
        "R-Precision": r_precision,
        "NDCG": ndcg,
        "Clicks": click
    }

    local_metrics = {
        "Precision": precisionk,
        "Recall": recallk,
        "MAP": average_precisionk
    }

    output = dict()
    local_metric_names = list(set(metric_names).intersection(local_metrics.keys()))
    local_results = {}
    for k in atK:
        hits_k=hits[:k]
        if pos_num>0:
            for name in local_metric_names:
                local_results[f'{name}@{k}']=local_metrics[name](true_num=pos_num,hits=hits_k)

    global_metric_names = list(set(metric_names).intersection(global_metrics.keys()))
    global_results = {}


    if pos_num > 0:
        for name in global_metric_names:
            global_results[name]=global_metrics[name](true_num=pos_num,hits=hits)

    output.update(local_results)
    output.update(global_results)
    return output


def evaluate_explanation(hits,pos_num, metric_names, atK):
    global_metrics = {
        # "R-Precision": r_precision,
        "NDCG": ndcg,
        "Precision": precisionk,
        "Recall": recallk,
        "MAP": average_precisionk
    }
    output = dict()
    global_metric_names = list(set(metric_names).intersection(global_metrics.keys()))
    for k in atK:
        results= dict()
        for name in global_metric_names:
            output[f'{name}@{k}'] =global_metrics[name](true_num=pos_num, hits=hits[:k])
    return output
