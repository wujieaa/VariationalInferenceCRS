from tqdm import tqdm

import numpy as np
import pandas as pd


def recallk(vector_true_dense, hits, **unused):
    hits = sum(hits)
    return float(hits)/len(vector_true_dense)


def precisionk(vector_predict, hits, **unused):
    hits = sum(hits)
    return float(hits)/len(vector_predict)


def average_precisionk(vector_predict, hits, **unused):
    precisions = np.cumsum(hits, dtype=np.float32)/range(1, len(vector_predict)+1)
    return np.mean(precisions)


def r_precision(vector_true_dense, vector_predict, **unused):
    vector_predict_short = vector_predict[:len(vector_true_dense)]
    hits = sum(np.isin(vector_predict_short, vector_true_dense))
    return float(hits)/len(vector_true_dense)


def _dcg_support(size):
    arr = np.arange(1, size+1)+1
    return 1./np.log2(arr)


def ndcg(vector_true_dense, vector_predict, hits):
    idcg = np.sum(_dcg_support(len(vector_true_dense)))
    dcg_base = _dcg_support(len(vector_predict))
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg/idcg


def click(hits, **unused):
    first_hit = next((i for i, x in enumerate(hits) if x), None)
    if first_hit is None:
        return 5
    else:
        return first_hit/10


def evaluate(matrix_Predict, matrix_Test, metric_names, atK):
    """
    :param matrix_Predict: Rating matrix for evaluation, prediction.
    :param matrix_Test: Rating matrix for evaluation, true labels.
    :param metric_names: Evaluation metrics
    :param atK: Top K retrieval
    :return:
    """
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

    num_users = matrix_Predict.shape[0]

    for k in atK:

        local_metric_names = list(set(metric_names).intersection(local_metrics.keys()))
        results = {name: [] for name in local_metric_names}
        topK_Predict = matrix_Predict[:, :k]

        for index in range(topK_Predict.shape[0]):
            vector_predict = topK_Predict[index]
            if len(vector_predict.nonzero()[0]) > 0:
                vector_true = matrix_Test[index]
                vector_true_dense = vector_true.nonzero()[0]
                hits = np.isin(vector_predict, vector_true_dense)

                if vector_true_dense.size > 0:
                    for name in local_metric_names:
                        results[name].append(local_metrics[name](vector_true_dense=vector_true_dense,
                                                                 vector_predict=vector_predict,
                                                                 hits=hits))

        results_summary = dict()
        for name in local_metric_names:
            results_summary['{0}@{1}'.format(name, k)]  = np.average(results[name])
        output.update(results_summary)

    global_metric_names = list(set(metric_names).intersection(global_metrics.keys()))
    results = {name: [] for name in global_metric_names}

    for index in tqdm(range(matrix_Predict.shape[0])):
        vector_predict = matrix_Predict[index,:]

        if len(vector_predict.nonzero()[0]) > 0:
            vector_true = matrix_Test[index]
            vector_true_dense = vector_true.nonzero()[0]
            hits = np.isin(vector_predict, vector_true_dense)

            # if user_index == 1:
            #     import ipdb;
            #     ipdb.set_trace()

            if vector_true_dense.size > 0:
                for name in global_metric_names:
                    results[name].append(global_metrics[name](vector_true_dense=vector_true_dense,
                                                              vector_predict=vector_predict,
                                                              hits=hits))

    results_summary = dict()
    for name in global_metric_names:
        results_summary[name] = np.average(results[name])
    output.update(results_summary)

    return output


def evaluate_explanation(df_predict, df_test, metric_names, atK, user_col,
                         item_col, rating_col, keyphrase_vector_col):
    df_test = df_test[(df_test[rating_col] == 1) & (df_test[keyphrase_vector_col] != '[]')] # Remove negatives reviews and reviews without keyphrases matched
    df_test = df_test[[user_col, item_col, keyphrase_vector_col]]
    df_test[keyphrase_vector_col] = df_test[keyphrase_vector_col].apply(lambda x: eval(x))
    df_predict = df_predict[[user_col, item_col, 'ExplanIndex']]
    res = pd.merge(df_test, df_predict, how='inner', on=[user_col, item_col])

    global_metrics = {
        # "R-Precision": r_precision,
        "NDCG": ndcg,
        "Precision": precisionk,
        "Recall": recallk,
        "MAP": average_precisionk
    }

    output = dict()

    num_interactionss = len(res)

    for k in atK:
        res['hits'] = res.apply(lambda x: list(np.isin(x['ExplanIndex'][:k], x[keyphrase_vector_col])), axis=1)

        global_metric_names = list(set(metric_names).intersection(global_metrics.keys()))

        for metric in tqdm(global_metric_names):
            res[metric] = res.apply(lambda x: global_metrics[metric](vector_true_dense=x[keyphrase_vector_col],
                                                                     vector_predict=x['ExplanIndex'][:k],
                                                                     hits=x['hits']),
                                    axis=1)
        results_summary = dict()
        for name in global_metric_names:
            results_summary['{0}@{1}'.format(name, k)] = (np.average(res[name]),
                                                          1.96 * np.std(res[name]) / np.sqrt(num_interactionss))
        output.update(results_summary)

    return output
