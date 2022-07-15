import pickle
from typing import List, Any

import torch

from torch.nn import functional as F
from torch import Tensor
from pytorch_lightning import LightningModule
from model.my_model4.ce_vi import CEVI
from metrics.evaluation import *

from model.my_model4.ce_vi2 import CEVI2
from model.my_model4.ce_vi3 import CEVI3
from model.my_model4.ce_vi4 import CEVI4
from model.my_model4.ce_vi5 import CEVI5
from model.my_model4.ce_vi6 import CEVI6
from utils.mylog import LogToFile
import os


class CEVILightning(LightningModule):
    '''
    负采样,prior
    '''
    FINAL_OUTPUTS_ITEM = 'item'
    FINAL_OUTPUTS_EXPLAINATION = 'explaination'
    FINAL_OUTPUTS_CRITIQUING = 'critiquing'
    FINAL_OUTPUTS_MINLOSS = 'minloss'
    FINAL_OUTPUTS_AVGLOSS = 'avgloss'
    FINAL_OUTPUTS_RATING_AVGLOSS = 'rating_avgloss'
    FINAL_OUTPUTS_KEYPHRASE_AVGLOSS = 'keyphrase_avgloss'

    def __init__(self,
                 *, word_num, user_num, item_num, args) -> None:
        super().__init__()
        self.args = args
        model_map={'CEVI':CEVI,'CEVI2':CEVI2,'CEVI3':CEVI3,'CEVI4':CEVI4,'CEVI5':CEVI5,'CEVI6':CEVI6}
        self.model = model_map[args.model_name](word_num=word_num, user_num=user_num, item_num=item_num, args=args)

        self.final_outputs = {
            CEVILightning.FINAL_OUTPUTS_ITEM: [],
            CEVILightning.FINAL_OUTPUTS_EXPLAINATION: [],
            CEVILightning.FINAL_OUTPUTS_CRITIQUING: [],
            CEVILightning.FINAL_OUTPUTS_MINLOSS: [],
            CEVILightning.FINAL_OUTPUTS_AVGLOSS: [],
            CEVILightning.FINAL_OUTPUTS_RATING_AVGLOSS: [],
            CEVILightning.FINAL_OUTPUTS_KEYPHRASE_AVGLOSS: [],
        }
        self.epoch_num = 0
        path = os.path.abspath(os.path.join(self.args.result_dir, 'lightning_logs/log.txt'))
        self.mylog = LogToFile(path)
        self.mylog.clear()
        self.user_num = user_num
        self.item_num = item_num
        self.word_num = word_num
        tmp_k = 5
        atk = []
        while tmp_k < self.args.topk:
            atk.append(tmp_k)
            tmp_k += 5
        atk.append(self.args.topk)
        self.atk = atk
        self.rating_metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
        self.explanation_metric_names = ['Recall', 'Precision']
        self.critiquing_keyphrase_topk_array = [5, 10, 20]

    def training_step(self, batch, batch_idx):
        user = batch[:, 0].long()
        item = batch[:, 1].long()
        rating_label = batch[:, 2]
        prior = batch[:, 3:]
        rating_scores, z = self.model(user, item)

        loss, rating_loss, keyphrase_loss = self.loss_function(rating_scores=rating_scores,
                                                               rating_label=rating_label,
                                                               z=z,
                                                               prior=prior)
        return {'loss': loss, 'rating_loss': rating_loss, 'keyphrase_loss': keyphrase_loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # loss
        loss_arr = list(map(lambda x: x['loss'].item(), outputs))
        minloss = min(loss_arr)
        avgloss = sum(loss_arr) / len(loss_arr)
        self.mylog.log(f'training epoch:{self.epoch_num},min loss:{minloss}')
        self.final_outputs[CEVILightning.FINAL_OUTPUTS_MINLOSS].append(minloss)
        self.final_outputs[CEVILightning.FINAL_OUTPUTS_AVGLOSS].append(avgloss)
        # rating loss
        loss_arr = list(map(lambda x: x['rating_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CEVILightning.FINAL_OUTPUTS_RATING_AVGLOSS].append(avgloss)
        # keyphrase_loss
        loss_arr = list(map(lambda x: x['keyphrase_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CEVILightning.FINAL_OUTPUTS_KEYPHRASE_AVGLOSS].append(avgloss)

    def validation_step(self, batch, batch_idx):
        batch = torch.squeeze(batch)
        user = batch[:, 0]
        item = batch[:, 1]
        rating_label = batch[:, 2]
        keyphrase_label = batch[:, 3:]
        rating, keyphrase= self.model(user, item, is_training=False)
        rating=rating.view(-1)
        rating_label=rating_label.view(-1)
        _, rating_topk_indices = torch.topk(rating, self.args.topk)
        pos_num = torch.sum(rating_label)
        hits = rating_label[rating_topk_indices]
        hits=hits.cpu().numpy()
        item_pre_outputs = self.evaluate_item_recommendation(hits, pos_num.item())

        pos_keyphrase = keyphrase[rating_label > 0, :]
        keyphrase_pre = torch.argsort(pos_keyphrase, dim=-1, descending=True)[:, :self.args.topk_keyphrase]
        pos_keyphrase_label = keyphrase_label[rating_label > 0, :]
        keyphrase_pre_np = keyphrase_pre.cpu().numpy()
        pos_keyphrase_label_np = pos_keyphrase_label.cpu().numpy()
        explanation_outputs = self.evaluate_explaination_recommendation(keyphrase_pre_np, pos_keyphrase_label_np)
        critiquing_outputs = self.evaluate_critiquing(user, item, keyphrase, keyphrase_pre,
                                                      rating_topk_indices)
        return [item_pre_outputs, explanation_outputs, critiquing_outputs]

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
            results = [o[1][f'{name}@{self.args.topk_keyphrase}'] for o in outputs]
            explanation_results_summary[f'{name}@{self.args.topk_keyphrase}'] = (np.average(results),
                                                          1.96 * np.std(results) / np.sqrt(len(results)))

        # -------evaluate Critiquing --------
        fmap_results_dict = dict()
        for k in self.critiquing_keyphrase_topk_array:
            results = [v for o in outputs for v in o[2][k]]
            results = np.array(results).flatten()
            if np.size(results) == 0:
                fmap_results_dict['F-MAP@{0}'.format(k)] = (np.nan, np.nan)
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
        self.final_outputs[CEVILightning.FINAL_OUTPUTS_ITEM].append(item_results_summary)
        self.final_outputs[CEVILightning.FINAL_OUTPUTS_EXPLAINATION].append(explanation_results_summary)
        self.final_outputs[CEVILightning.FINAL_OUTPUTS_CRITIQUING].append(fmap_results_dict)

    def on_fit_end(self):
        # save result
        result_path = os.path.abspath(os.path.join(self.args.result_dir, f'lightning_logs/{self.args.result_file_name}.pkl'))
        if (os.path.exists(result_path)):
            os.remove(result_path)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'wb') as f:
            pickle.dump(self.final_outputs, f, pickle.HIGHEST_PROTOCOL)

        # log
        ndcgs = map(lambda a: a['NDCG'][0], self.final_outputs[CEVILightning.FINAL_OUTPUTS_ITEM])
        max_ndcg = max(ndcgs)
        self.mylog.log('final result:',
                       f'max_ndcg:{max_ndcg}',
                       'done!')

    def evaluate_critiquing(self, user, item, keyphrase, keyphrase_pre, rating_topk_indices):
        unique_keyphrase = torch.unique(keyphrase_pre)
        keyphrase_indexes = unique_keyphrase[torch.randperm(unique_keyphrase.shape[0])[:5]]
        rating_topk_item_np = item[rating_topk_indices].cpu().numpy()
        fmap_results = {k: [] for k in self.critiquing_keyphrase_topk_array}
        for critiqued_keyphrase_index in keyphrase_indexes:
            indecies = torch.where(keyphrase_pre == critiqued_keyphrase_index)[0]
            affected_items_np = item[indecies].cpu().numpy()

            critiquing_rating = self.model.looping_predict(keyphrase,critiqued_keyphrase_index.item(),user)
            _, modified_rating_topk_indices = torch.topk(critiquing_rating.squeeze(), self.args.topk)
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
            pos_num=len(item_label)
            if pos_num>0:
                explanation_result.append(evaluate_explanation(hits,
                                                           pos_num,
                                                           self.explanation_metric_names,
                                                           [self.args.topk_keyphrase]
                                                           ))
        outputs = {}
        for name in self.explanation_metric_names:
            arr = [r[f'{name}@{self.args.topk_keyphrase}'] for r in explanation_result]
            outputs[f'{name}@{self.args.topk_keyphrase}'] = np.average(arr)
        return outputs

    def evaluate_item_recommendation(self, hits, pos_num):
        item_pre_outputs = evaluate(hits, pos_num, self.rating_metric_names, self.atk, analytical=False)
        return item_pre_outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2_weight)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,40, 80], gamma=0.1)
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler
                }

        # return optimizer

    def forward(self, input: Tensor, **kwargs) -> tuple:
        user = input[:, 0]
        item = input[:, 1]
        return self.model(user, item)

    def loss_function(self, *, rating_scores, rating_label,z, prior):
        rating_scores = rating_scores.view(-1)
        rating_loss = F.binary_cross_entropy_with_logits(rating_scores, rating_label)
        p=prior
        log_q = torch.log(z)
        log_p=torch.log(p)
        neg_log_q = torch.log(1-z)
        neg_log_p = torch.log(1-p)
        kld = torch.mean(p * (log_p - log_q) + (1 - p) * (neg_log_p - neg_log_q))
        loss = self.args.rating_weight * rating_loss + self.args.keyphrase_weight * kld

        return loss, rating_loss, kld

