import pickle
from typing import List, Any

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch import Tensor
from pytorch_lightning import LightningModule
from model.ce_ncf.ce_ncf import CENCF
from model.ce_ncf.params import *
import pandas as pd
import scipy.sparse as sparse
from metrics.evaluation import *
import time
from utils.mylog import LogToFile
from data_model.data_const import *
import os
from functools import reduce


class CENCFLightning(LightningModule):
    FINAL_OUTPUTS_ITEM = 'item'
    FINAL_OUTPUTS_EXPLAINATION = 'explaination'
    FINAL_OUTPUTS_CRITIQUING = 'critiquing'
    FINAL_OUTPUTS_MINLOSS = 'minloss'
    FINAL_OUTPUTS_AVGLOSS = 'avgloss'
    FINAL_OUTPUTS_RATING_AVGLOSS = 'rating_avgloss'
    FINAL_OUTPUTS_KEYPHRASE_AVGLOSS = 'keyphrase_avgloss'
    FINAL_OUTPUTS_RECON_AVGLOSS = 'recon_avgloss'

    def __init__(self,
                 *, word_num, user_num, item_num) -> None:
        super().__init__()
        self.model = CENCF(word_num=word_num, user_num=user_num, item_num=item_num)

        self.final_outputs = {
            CENCFLightning.FINAL_OUTPUTS_ITEM: [],
            CENCFLightning.FINAL_OUTPUTS_EXPLAINATION: [],
            CENCFLightning.FINAL_OUTPUTS_CRITIQUING: [],
            CENCFLightning.FINAL_OUTPUTS_MINLOSS: [],
            CENCFLightning.FINAL_OUTPUTS_AVGLOSS: [],
            CENCFLightning.FINAL_OUTPUTS_RATING_AVGLOSS: [],
            CENCFLightning.FINAL_OUTPUTS_KEYPHRASE_AVGLOSS: [],
            CENCFLightning.FINAL_OUTPUTS_RECON_AVGLOSS: []
        }
        self.epoch_num = 0
        path = os.path.join(os.path.abspath('../../lightning_logs'), 'log.txt')
        self.mylog = LogToFile(path)
        self.mylog.clear()
        self.user_num = user_num
        self.item_num = item_num
        self.word_num = word_num
        tmp_k = 5
        atk = []
        while tmp_k < topk:
            atk.append(tmp_k)
            tmp_k += 5
        atk.append(topk)
        self.atk = atk
        self.rating_metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
        self.explanation_metric_names = ['Recall', 'Precision']
        self.critiquing_keyphrase_topk_array = [5, 10, 20]

    def training_step(self, batch, batch_idx):
        user = batch[:, 0]
        item = batch[:, 1]
        rating_label = batch[:, 2].float()
        keyphrase_label = batch[:, 3:].float()
        rating, keyphrase, reconstructed_latent, detached_latent = self.model(user, item)
        loss, rating_loss, keyphrase_loss, recons_loss = self.loss_function(rating=rating, rating_label=rating_label,
                                                                            keyphrase=keyphrase,
                                                                            keyphrase_label=keyphrase_label,
                                                                            reconstructed_latent=reconstructed_latent,
                                                                            detached_latent=detached_latent)
        return {'loss': loss, 'rating_loss': rating_loss, 'keyphrase_loss': keyphrase_loss, 'recons_loss': recons_loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # loss
        loss_arr = list(map(lambda x: x['loss'].item(), outputs))
        minloss = min(loss_arr)
        avgloss = sum(loss_arr) / len(loss_arr)
        self.mylog.log(f'training epoch:{self.epoch_num},min loss:{minloss}')
        self.final_outputs[CENCFLightning.FINAL_OUTPUTS_MINLOSS].append(minloss)
        self.final_outputs[CENCFLightning.FINAL_OUTPUTS_AVGLOSS].append(avgloss)
        # rating loss
        loss_arr = list(map(lambda x: x['rating_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CENCFLightning.FINAL_OUTPUTS_RATING_AVGLOSS].append(avgloss)
        # keyphrase_loss
        loss_arr = list(map(lambda x: x['keyphrase_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CENCFLightning.FINAL_OUTPUTS_KEYPHRASE_AVGLOSS].append(avgloss)
        # recons_loss
        loss_arr = list(map(lambda x: x['recons_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CENCFLightning.FINAL_OUTPUTS_RECON_AVGLOSS].append(avgloss)

    def validation_step(self, batch, batch_idx):
        batch = torch.squeeze(batch)
        user = batch[:, 0]
        item = batch[:, 1]
        rating_label = batch[:, 2]
        keyphrase_label = batch[:, 3:]
        rating, keyphrase, reconstructed_latent, detached_latent = self.model(user, item)
        # prediction = torch.cat([batch, rating,keyphrase], axis=1)
        _, rating_topk_indices = torch.topk(rating.squeeze(), topk)
        pos_num = torch.sum(rating_label)
        hits = rating_label[rating_topk_indices]
        item_pre_outputs = self.evaluate_item_recommendation(hits.cpu().numpy(), pos_num.item())

        pos_keyphrase = keyphrase[rating_label > 0, :]
        keyphrase_pre = torch.argsort(pos_keyphrase, dim=-1, descending=True)[:, :topk_keyphrase]
        pos_keyphrase_label = keyphrase_label[rating_label > 0, :]
        keyphrase_pre_np = keyphrase_pre.cpu().numpy()
        pos_keyphrase_label_np = pos_keyphrase_label.cpu().numpy()
        explanation_outputs = self.evaluate_explaination_recommendation(keyphrase_pre_np, pos_keyphrase_label_np)
        critiquing_outputs = self.evaluate_critiquing(detached_latent, item, keyphrase, keyphrase_pre,
                                                      rating_topk_indices)
        return [item_pre_outputs, explanation_outputs,critiquing_outputs]



    def validation_epoch_end(self, outputs) -> None:
        # ------evaluate item recommendation --------
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
        item_results_summary = dict()
        local_metric_names = list(set(self.rating_metric_names).intersection(local_metrics.keys()))
        global_metric_names = list(set(self.rating_metric_names).intersection(global_metrics.keys()))
        local_item_results_summary = dict()
        for k in self.atk:
            for name in local_metric_names:
                results = [o[0][f'{name}@{k}'] for o in outputs]
                local_item_results_summary[f'{name}@{k}'] = (np.average(results),
                                                             1.96 * np.std(results) / np.sqrt(len(results)))
        item_results_summary.update(local_item_results_summary)
        global_item_results_summary = dict()
        for name in global_metric_names:
            results = [o[0][name] for o in outputs]
            global_item_results_summary[name] = (np.average(results),
                                                 1.96 * np.std(results) / np.sqrt(len(results)))
        item_results_summary.update(global_item_results_summary)
        # ------evaluate explaination recommendation --------
        explanation_results_summary = dict()
        for name in self.explanation_metric_names:
            results = [o[1][f'{name}@{topk_keyphrase}'] for o in outputs]
            explanation_results_summary[f'{name}@{topk_keyphrase}'] = (np.average(results),
                                                          1.96 * np.std(results) / np.sqrt(len(results)))

        # -------evaluate Critiquing --------
        fmap_results_dict=dict()
        for k in self.critiquing_keyphrase_topk_array:
            results=[v for o in outputs for v in o[2][k]]
            results=np.array(results).flatten()
            if np.size(results) ==0 :
                fmap_results_dict['F-MAP@{0}'.format(k)]=(np.nan,np.nan)
            else:
                fmap_results_dict['F-MAP@{0}'.format(k)] = (np.average(results),
                                                        1.96 * np.std(results) / np.sqrt(
                                                            np.sqrt(len(results))))
        # -----output results
        self.mylog.log(f'epoch:{self.epoch_num}',
                       'item evaluation',
                       str(item_results_summary),
                       'explanation evaluation',
                       str(explanation_results_summary),
                       'critiquing evaluation',
                       str(fmap_results_dict)
                       )
        self.print(item_results_summary)
        self.print(explanation_results_summary)
        self.print(fmap_results_dict)
        self.log('val_acc', item_results_summary['NDCG'][0])  # val_acc自动取最大，val_loss自动取最小
        self.epoch_num += 1
        self.final_outputs[CENCFLightning.FINAL_OUTPUTS_ITEM].append(item_results_summary)
        self.final_outputs[CENCFLightning.FINAL_OUTPUTS_EXPLAINATION].append(explanation_results_summary)
        self.final_outputs[CENCFLightning.FINAL_OUTPUTS_CRITIQUING].append(fmap_results_dict)

    def on_fit_end(self):
        # save result
        result_path = os.path.abspath('../../lightning_logs/cencf_result.pkl')
        if (os.path.exists(result_path)):
            os.remove(result_path)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'wb') as f:
            pickle.dump(self.final_outputs, f, pickle.HIGHEST_PROTOCOL)

        # log
        ndcgs = map(lambda a: a['NDCG'][0], self.final_outputs[CENCFLightning.FINAL_OUTPUTS_ITEM])
        max_ndcg = max(ndcgs)
        self.mylog.log('final result:',
                       f'max_ndcg:{max_ndcg}',
                       'done!')


    def evaluate_critiquing(self, detached_latent, item, keyphrase, keyphrase_pre, rating_topk_indices):
        unique_keyphrase = torch.unique(keyphrase_pre)
        keyphrase_indexes = unique_keyphrase[torch.randperm(unique_keyphrase.shape[0])[:5]]
        rating_topk_item_np = item[rating_topk_indices].cpu().numpy()
        fmap_results = {k: [] for k in self.critiquing_keyphrase_topk_array}
        for key_phrase_index in keyphrase_indexes:
            indecies = torch.where(keyphrase_pre == key_phrase_index)[0]
            affected_items_np = item[indecies].cpu().numpy()
            modified_keyhprase = torch.clone(keyphrase)
            min_val=torch.min(keyphrase,dim=1).values
            modified_keyhprase[:, key_phrase_index.item()] = min_val
            critiquing_rating, critiquing_keyphrase = self.model.looping_predict(modified_keyhprase, detached_latent)
            _, modified_rating_topk_indices = torch.topk(critiquing_rating.squeeze(), topk)
            modified_rating_topk_item_np = item[modified_rating_topk_indices].cpu().numpy()
            for k in self.critiquing_keyphrase_topk_array:
                if np.any(np.isin(rating_topk_item_np[:k], affected_items_np)):
                        fmap_results[k].append(average_precisionk(np.isin(rating_topk_item_np[:k], affected_items_np))
                                               - average_precisionk(np.isin(modified_rating_topk_item_np[:k], affected_items_np)))
        return fmap_results
    def evaluate_explaination_recommendation(self, keyphrase_pre, pos_keyphrase_label):
        explanation_result = []
        for i in range(keyphrase_pre.shape[0]):
            item_topk = keyphrase_pre[i]
            item_label = np.nonzero(pos_keyphrase_label[i])[0]
            hits = np.isin(item_topk, item_label)
            pos_num = len(item_label)
            if pos_num > 0:
                explanation_result.append(evaluate_explanation(hits,
                                                               pos_num,
                                                               self.explanation_metric_names,
                                                               [self.args.topk_keyphrase]
                                                               ))
        outputs = {}
        for name in self.explanation_metric_names:
            arr = [r[f'{name}@{topk_keyphrase}'] for r in explanation_result]
            outputs[f'{name}@{topk_keyphrase}'] = np.average(arr)
        return outputs

    def evaluate_item_recommendation(self, hits, pos_num):
        item_pre_outputs = evaluate(hits, pos_num, self.rating_metric_names, self.atk, analytical=False)
        return item_pre_outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=l2_weight)
        return optimizer

    def forward(self, input: Tensor, **kwargs) -> tuple:
        user = input[:, 0]
        item = input[:, 1]
        return self.model(user, item)

    def loss_function(self, *, rating, rating_label, keyphrase, keyphrase_label, reconstructed_latent,
                      detached_latent) -> float:
        rating_loss = F.mse_loss(rating.view(-1), rating_label.view(-1))
        keyphrase_loss = F.mse_loss(keyphrase, keyphrase_label)
        recons_loss = F.mse_loss(reconstructed_latent, detached_latent)
        loss = rating_weight * rating_loss + keyphrase_weight * keyphrase_loss + recons_weight * recons_loss
        return loss, rating_loss, keyphrase_loss, recons_loss

    def looping_predict(self, *, modified_keyphrase, old_latent):
        rating, keyphrase = self.model.looping_predict(modified_keyphrase=modified_keyphrase, old_latent=old_latent)
        return rating, keyphrase

    def critique_keyphrase(self, user_index, num_items, topk_keyphrase=10):
        # Get the given user and all item pairs as input to critiquing models
        items = torch.arange(num_items, dtype=torch.int64)
        users =torch.full_like(items,user_index, dtype=torch.int64)
        if torch.cuda.is_available():
            items = items.cuda()
            users = users.cuda()
        # Get rating and explanation prediction for the given user and all item pairs
        rating, keyphrase, _, old_latent = self.model(users, items)
        rating, keyphrase = rating.cpu().numpy(), keyphrase.cpu().numpy()
        # For the given user, get top k keyphrases for each item
        explanation_rank_list = np.argsort(-keyphrase, axis=1)[:, :topk_keyphrase]

        # Random critique one keyphrase among existing predicted keyphrases
        unique_keyphrase = np.unique(explanation_rank_list)
        keyphrase_index = int(np.random.choice(unique_keyphrase, 1)[0])

        # Get all affected items
        affected_items = np.where(explanation_rank_list == keyphrase_index)[0]

        # Zero out the critiqued keyphrase in all items
        minval = torch.tensor(np.min(keyphrase, axis=1))
        keyphrase[:, keyphrase_index] = minval
        keyphrase = torch.tensor(keyphrase)
        if torch.cuda.is_available():
            keyphrase = keyphrase.cuda()
        modified_rating, modified_explanation = self.looping_predict(modified_keyphrase=keyphrase,
                                                                     old_latent=old_latent)
        modified_rating, modified_explanation = modified_rating.cpu().numpy(), modified_explanation.cpu().numpy()
        return np.argsort(rating.flatten())[::-1], np.argsort(modified_rating.flatten())[::-1], affected_items
