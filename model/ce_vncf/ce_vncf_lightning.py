import pickle
from typing import List, Any

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch import Tensor
from pytorch_lightning import LightningModule
from model.ce_vncf.ce_vncf_origin import CEVNCFOrigin
from model.ce_vncf.params import *
import pandas as pd
import scipy.sparse as sparse
from metrics.evaluation import *
import time
from utils.mylog import LogToFile
from data_model.data_const import *
import os
from functools import reduce

class CEVNCFLightning(LightningModule):
    FINAL_OUTPUTS_ITEM='item'
    FINAL_OUTPUTS_EXPLAINATION='explaination'
    FINAL_OUTPUTS_CRITIQUING='critiquing'
    FINAL_OUTPUTS_MINLOSS='minloss'
    FINAL_OUTPUTS_AVGLOSS='avgloss'
    FINAL_OUTPUTS_RATING_AVGLOSS = 'rating_avgloss'
    FINAL_OUTPUTS_KEYPHRASE_AVGLOSS = 'keyphrase_avgloss'
    FINAL_OUTPUTS_RECON_AVGLOSS = 'recon_avgloss'
    def __init__(self,
                 *,test_df, word_num,user_num, item_num) -> None:
        super().__init__()
        self.model=CEVNCFOrigin(word_num=word_num, user_num=user_num, item_num=item_num)
        test_df.loc[:, RATING_ID]=1
        self.test_df=test_df[[USER_INDEX, ITEM_INDEX, RATING_ID, KEY_VECTOR]]

        self.R_valid = self.to_sparse_matrix(test_df,
                                             user_num,
                                             item_num,
                                             USER_INDEX,
                                             ITEM_INDEX,
                                             RATING_ID)
        self.final_outputs={
            CEVNCFLightning.FINAL_OUTPUTS_ITEM:[],
            CEVNCFLightning.FINAL_OUTPUTS_EXPLAINATION:[],
            CEVNCFLightning.FINAL_OUTPUTS_CRITIQUING:[],
            CEVNCFLightning.FINAL_OUTPUTS_MINLOSS:[],
            CEVNCFLightning.FINAL_OUTPUTS_AVGLOSS:[],
            CEVNCFLightning.FINAL_OUTPUTS_RATING_AVGLOSS:[],
            CEVNCFLightning.FINAL_OUTPUTS_KEYPHRASE_AVGLOSS:[],
            CEVNCFLightning.FINAL_OUTPUTS_RECON_AVGLOSS:[]
        }
        self.epoch_num=0
        path=os.path.join(os.path.abspath('../../lightning_logs'),'log.txt')
        self.mylog=LogToFile(path)
        self.mylog.clear()
        self.user_num=user_num
        self.item_num=item_num
        self.word_num=word_num

    def to_sparse_matrix(self,df, num_user, num_item, user_col, item_col, rating_col):
        dok = df[[user_col, item_col, rating_col]].copy()
        dok = dok.values
        dok = dok[dok[:, 2] > 0]
        shape = (num_user, num_item)

        return sparse.csr_matrix((dok[:, 2].astype(np.float32), (dok[:, 0], dok[:, 1])), shape=shape)

    def training_step(self, batch, batch_idx):
        user = batch[0]
        item = batch[1]
        rating_label=batch[2]
        keyphrase_label = batch[3]
        rating,keyphrase,reconstructed_latent,detached_latent,mean,logvar=self.model(user,item)
        loss,rating_loss,keyphrase_loss,recons_loss=self.loss_function(rating=rating,
                                                                       rating_label=rating_label,
                                                                       keyphrase=keyphrase,
                                                                       keyphrase_label=keyphrase_label,
                                                                       reconstructed_latent=reconstructed_latent,
                                                                       detached_latent=detached_latent,
                                                                       mean=mean,
                                                                       logvar=logvar)
        return {'loss': loss, 'rating_loss': rating_loss,'keyphrase_loss':keyphrase_loss,'recons_loss':recons_loss}
    def training_epoch_end(self, outputs: List[Any]) -> None:
        #loss
        loss_arr=list(map(lambda x: x['loss'].item(), outputs))
        minloss=min(loss_arr)
        avgloss=sum(loss_arr)/len(loss_arr)
        self.mylog.log(f'training epoch:{self.epoch_num},min loss:{minloss}')
        self.final_outputs[CEVNCFLightning.FINAL_OUTPUTS_MINLOSS].append(minloss)
        self.final_outputs[CEVNCFLightning.FINAL_OUTPUTS_AVGLOSS].append(avgloss)
        #rating loss
        loss_arr = list(map(lambda x: x['rating_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CEVNCFLightning.FINAL_OUTPUTS_RATING_AVGLOSS].append(avgloss)
        #keyphrase_loss
        loss_arr = list(map(lambda x: x['keyphrase_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CEVNCFLightning.FINAL_OUTPUTS_KEYPHRASE_AVGLOSS].append(avgloss)
        #recons_loss
        loss_arr = list(map(lambda x: x['recons_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CEVNCFLightning.FINAL_OUTPUTS_RECON_AVGLOSS].append(avgloss)

    def validation_step(self,batch,batch_idx):
        return self.evaluate_step(batch, batch_idx)
        # if loader=='test_data_neg':
        #     return self.evaluate_step(batch,batch_idx)
        # elif loader=='test_data':
        #     return self.evaluate_explaination_step(batch,batch_idx)
        # else:
        #     raise Exception(f'loader value:{loader} is not defined')
    def evaluate_step(self,batch,batch_idx):
        user_id=batch[0,0].item()
        items=batch[0,1:].nonzero(as_tuple=False)
        items=items.view(-1)  # 此处没有行坐标
        users=torch.empty_like(items,dtype=torch.int64)
        users[:]=user_id
        ratings, _, _, _,_,_ = self.model(users, items,sampling=True)
        prediction=np.concatenate([items.view(-1,1).cpu().numpy(),ratings.cpu().numpy()],axis=1)
        prediction = prediction[prediction[:, 1].argsort()[::-1][:topk]]
        candidates = prediction[:, 0].astype(int).flatten()
        user_candidate_items = np.insert(candidates, 0, user_id)
        return user_candidate_items


    def validation_epoch_end(self, outputs) -> None:
        user_candidate_items=outputs
        # ------evaluate item recommendation --------
        item_pre_outputs = self.evaluate_item_recommendation(user_candidate_items)
        # ------evaluate explaination recommendation --------
        explanation_result = self.evaluate_explaination_recommendation()
        #-------evaluate Critiquing --------
        critiquing_result = self.evaluate_critiquing()
        #-----output results
        self.mylog.log(f'epoch:{self.epoch_num}',
                       'item evaluation',
                       str(item_pre_outputs),
                       'explanation evaluation',
                       str(explanation_result),
                       'critiquing evaluation',
                       str(critiquing_result)
                       )
        self.print(item_pre_outputs)
        self.print(explanation_result)
        self.print(critiquing_result)
        self.log('val_acc', item_pre_outputs['NDCG'][0])  # val_acc自动取最大，val_loss自动取最小
        self.epoch_num += 1
        self.final_outputs[CEVNCFLightning.FINAL_OUTPUTS_ITEM].append(item_pre_outputs)
        self.final_outputs[CEVNCFLightning.FINAL_OUTPUTS_EXPLAINATION].append(explanation_result)
        self.final_outputs[CEVNCFLightning.FINAL_OUTPUTS_CRITIQUING].append(critiquing_result)
    def on_fit_end(self):
        #save result
        result_path = os.path.abspath('../../lightning_logs/result.pkl')
        if (os.path.exists(result_path)):
            os.remove(result_path)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'wb') as f:
            pickle.dump(self.final_outputs, f, pickle.HIGHEST_PROTOCOL)

        #log
        ndcgs=map(lambda a: a['NDCG'][0], self.final_outputs[CEVNCFLightning.FINAL_OUTPUTS_ITEM])
        max_ndcg=max(ndcgs)
        self.mylog.log('final result:',
                       f'max_ndcg:{max_ndcg}',
                       'done!')

    def evaluate_critiquing(self,analytical=False):
        keyphrase_topk_array = [5, 10, 20]
        fmap_results = [[] for _ in keyphrase_topk_array]
        for iteration in range(3):
            sampled_users = np.random.choice(self.user_num, num_sampled_users)
            for user in tqdm(sampled_users):

                top_items_before_critique, top_items_after_critique, affected_items \
                    = self.critique_keyphrase(user, self.item_num, topk_keyphrase=topk_keyphrase)

                # all_items = np.array(range(self.num_items))
                # unaffected_items = all_items[~np.in1d(all_items, affected_items)]

                for i, k in enumerate(keyphrase_topk_array):
                    fmap_results[i].append(average_precisionk(top_items_before_critique[:k],
                                                              np.isin(top_items_before_critique[:k], affected_items))
                                           - average_precisionk(top_items_after_critique[:k],
                                                                np.isin(top_items_after_critique[:k], affected_items)))
        fmap_results_dict = dict()
        for i, k in enumerate(keyphrase_topk_array):
            if analytical:
                fmap_results_dict['F-MAP@{0}'.format(k)] = fmap_results[i]
            else:
                fmap_results_dict['F-MAP@{0}'.format(k)] = (np.average(fmap_results[i]),
                                                                  1.96 * np.std(fmap_results[i]) / np.sqrt(num_sampled_users*3))


        return fmap_results_dict

    def evaluate_explaination_recommendation(self):
        explanation = []
        users = torch.tensor(self.test_df[USER_INDEX].values, dtype=torch.int64)
        items = torch.tensor(self.test_df[ITEM_INDEX].values, dtype=torch.int64)
        if torch.cuda.is_available():
            users = users.cuda()
            items = items.cuda()
        _, keyphrases, _, _,_,_ = self.model(users, items,sampling=False)
        keyphrases=keyphrases.cpu().numpy()
        for explanation_score in tqdm(keyphrases):
            explanation.append(np.argsort(explanation_score)[::-1][:topk_keyphrase])
        df_predicted_explanation = pd.DataFrame.from_dict({USER_INDEX: self.test_df[USER_INDEX].values,
                                                           ITEM_INDEX: self.test_df[ITEM_INDEX].values,
                                                           'ExplanIndex': explanation})
        explanation_result = evaluate_explanation(df_predicted_explanation,
                                                  self.test_df,
                                                  ['Recall', 'Precision'],
                                                  [topk_keyphrase],
                                                  USER_INDEX,
                                                  ITEM_INDEX,
                                                  RATING_ID,
                                                  KEY_VECTOR)
        return explanation_result

    def evaluate_item_recommendation(self, user_candidate_items):
        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
        matrix_Predict = np.vstack(user_candidate_items)
        tmpK = 5
        atk = []
        while tmpK < topk:
            atk.append(tmpK)
            tmpK += 5
        atk.append(topk)
        item_pre_outputs = evaluate(matrix_Predict, self.R_valid, metric_names, atk, analytical=False)
        return item_pre_outputs



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,weight_decay=l2_weight)
        return optimizer

    def forward(self, input: Tensor, **kwargs) -> tuple:
        user=input[:,0]
        item=input[:,1]
        return  self.model(user,item)

    def loss_function(self, *, rating, rating_label, keyphrase, keyphrase_label, reconstructed_latent, detached_latent,
                      mean,logvar):
        rating_loss = F.mse_loss(rating.view(-1), rating_label.view(-1))
        keyphrase_loss=F.mse_loss(keyphrase,keyphrase_label)
        recons_loss=F.mse_loss(reconstructed_latent,detached_latent)
        kl_loss = torch.mean(0.5 * torch.sum(torch.square(mean) + torch.exp(logvar) - logvar - 1,dim=1),dim=0)
        loss = rating_weight * rating_loss \
               + keyphrase_weight * keyphrase_loss \
               + recons_weight * recons_loss\
               + klloss_weight * kl_loss
        return loss,rating_loss,keyphrase_loss,recons_loss

    def looping_predict(self,*,modified_keyphrase,old_latent):
        rating,keyphrase=self.model.looping_predict(modified_keyphrase=modified_keyphrase,old_latent=old_latent)
        return rating,keyphrase

    def critique_keyphrase(self, user_index, num_items, topk_keyphrase=10):
        # Get the given user and all item pairs as input to critiquing models
        items=torch.arange(num_items,dtype=torch.int64)
        users = torch.empty_like(items, dtype=torch.int64)
        users[:] = user_index
        if torch.cuda.is_available():
            items=items.cuda()
            users=users.cuda()
        # Get rating and explanation prediction for the given user and all item pairs
        rating,keyphrase,_,old_latent,_,_ = self.model(users,items,sampling=False)
        rating, keyphrase=rating.cpu().numpy(),keyphrase.cpu().numpy()
        # For the given user, get top k keyphrases for each item
        explanation_rank_list = np.argsort(-keyphrase, axis=1)[:, :topk_keyphrase]

        # Random critique one keyphrase among existing predicted keyphrases
        unique_keyphrase = np.unique(explanation_rank_list)
        keyphrase_index = int(np.random.choice(unique_keyphrase, 1)[0])

        # Get all affected items
        affected_items = np.where(explanation_rank_list == keyphrase_index)[0]

        # Zero out the critiqued keyphrase in all items
        minval=torch.tensor(np.min(keyphrase, axis=1))
        keyphrase[:, keyphrase_index] = minval
        keyphrase = torch.tensor(keyphrase)
        if torch.cuda.is_available():
            keyphrase=keyphrase.cuda()
        modified_rating, modified_explanation = self.looping_predict(modified_keyphrase=keyphrase, old_latent=old_latent)
        modified_rating, modified_explanation=modified_rating.cpu().numpy(), modified_explanation.cpu().numpy()
        return np.argsort(rating.flatten())[::-1], np.argsort(modified_rating.flatten())[::-1], affected_items

