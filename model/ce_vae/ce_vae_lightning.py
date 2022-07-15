import pickle
from typing import List, Any

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch import Tensor
from pytorch_lightning import LightningModule

from data_model.beer_vae_data_module import BeerVaeDataModule
from model.ce_vae.ce_vae import CEVAE
from model.ce_vae.params import *
import pandas as pd
import scipy.sparse as sparse
from metrics.ave_evaluation import *
import time
from utils.mylog import LogToFile
from data_model.data_const import *
import os
from functools import reduce
from model.ce_vae.vae_predictor import *
import os.path as path
from ast import literal_eval
class CEVAELightning(LightningModule):
    FINAL_OUTPUTS_ITEM='item'
    FINAL_OUTPUTS_EXPLAINATION='explaination'
    FINAL_OUTPUTS_CRITIQUING='critiquing'
    FINAL_OUTPUTS_MINLOSS='minloss'
    FINAL_OUTPUTS_AVGLOSS='avgloss'
    FINAL_OUTPUTS_RATING_AVGLOSS = 'rating_avgloss'
    FINAL_OUTPUTS_KEYPHRASE_AVGLOSS = 'keyphrase_avgloss'
    FINAL_OUTPUTS_RECON_AVGLOSS = 'recon_avgloss'
    def __init__(self,
                 *, word_num, item_num) -> None:
        super().__init__()
        self.model=CEVAE(word_num=word_num,  item_num=item_num)

        self.final_outputs={
            CEVAELightning.FINAL_OUTPUTS_ITEM:[],
            CEVAELightning.FINAL_OUTPUTS_EXPLAINATION:[],
            CEVAELightning.FINAL_OUTPUTS_CRITIQUING:[],
            CEVAELightning.FINAL_OUTPUTS_MINLOSS:[],
            CEVAELightning.FINAL_OUTPUTS_AVGLOSS:[],
            CEVAELightning.FINAL_OUTPUTS_RATING_AVGLOSS:[],
            CEVAELightning.FINAL_OUTPUTS_KEYPHRASE_AVGLOSS:[],
            CEVAELightning.FINAL_OUTPUTS_RECON_AVGLOSS:[]
        }
        self.epoch_num=0
        logfile=os.path.join(os.path.abspath('../../lightning_logs'),'log.txt')
        self.mylog=LogToFile(logfile)
        self.mylog.clear()
        self.item_num=item_num
        self.word_num=word_num

    def prepare_data(self) -> None:
        self.train_df = pd.read_csv(path.join(data_dir, 'Train.csv'))
        self.train_df = self.train_df[[USER_INDEX, ITEM_INDEX, KEY_VECTOR]]
        self.train_df[KEY_VECTOR] = self.train_df[KEY_VECTOR].apply(literal_eval)
        user_num=self.train_df[USER_INDEX].nunique()
        u_k_i_map={(u,k):[] for k in range(self.word_num) for u in range(user_num)}
        for index,row in tqdm(self.train_df.iterrows()):
            u=row[USER_INDEX]
            item=row[ITEM_INDEX]
            k_list=row[KEY_VECTOR]
            for k in k_list:
                u_k_i_map[(u,k)].append(item)
        self.u_k_i_map=u_k_i_map

    def training_step(self, batch, batch_idx):
        items = batch[:,:self.item_num]
        keyphrase_label = batch[:,self.item_num:]
        rating,keyphrase,reconstructed_latent,detached_latent,mean,logvar=self.model(items)
        loss,rating_loss,keyphrase_loss,recons_loss=self.loss_function(rating=rating,
                                                                       rating_label=items,
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
        self.final_outputs[CEVAELightning.FINAL_OUTPUTS_MINLOSS].append(minloss)
        self.final_outputs[CEVAELightning.FINAL_OUTPUTS_AVGLOSS].append(avgloss)
        #rating loss
        loss_arr = list(map(lambda x: x['rating_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CEVAELightning.FINAL_OUTPUTS_RATING_AVGLOSS].append(avgloss)
        #keyphrase_loss
        loss_arr = list(map(lambda x: x['keyphrase_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CEVAELightning.FINAL_OUTPUTS_KEYPHRASE_AVGLOSS].append(avgloss)
        #recons_loss
        loss_arr = list(map(lambda x: x['recons_loss'].item(), outputs))
        avgloss = sum(loss_arr) / len(loss_arr)
        self.final_outputs[CEVAELightning.FINAL_OUTPUTS_RECON_AVGLOSS].append(avgloss)

    def validation_step(self,batch,batch_idx,dataloader_idx):

        if dataloader_idx==0:
            result = self.general_validation(batch)
        elif dataloader_idx==1: # -------evaluate Critiquing --------
            result = self.critiquing_validation(batch)
        else:
            raise Exception('dataloader_idx error')
        return result
    def validation_epoch_end(self, outputs) -> None:
        general_outputs=outputs[0]
        critiquing_outputs=outputs[1]
        # ------evaluate item recommendation --------
        item_pre_outputs = self.total_eval(general_outputs,'item')
        # ------evaluate explaination recommendation --------
        explanation_pre_outputs = self.total_eval(general_outputs,'explanation')
        #-------evaluate Critiquing --------
        critiquing_pre_outputs = self.total_eval(critiquing_outputs,'critiquing')
        #-----output results
        self.mylog.log(f'epoch:{self.epoch_num}',
                       'item evaluation',
                       str(item_pre_outputs),
                       'explanation evaluation',
                       str(explanation_pre_outputs),
                       'critiquing evaluation',
                       str(critiquing_pre_outputs)
                       )
        self.print(item_pre_outputs)
        self.print(explanation_pre_outputs)
        self.print(critiquing_pre_outputs)
        self.log('val_acc', item_pre_outputs['NDCG'])  # val_acc自动取最大，val_loss自动取最小
        self.epoch_num += 1
        self.final_outputs[CEVAELightning.FINAL_OUTPUTS_ITEM].append(item_pre_outputs)
        self.final_outputs[CEVAELightning.FINAL_OUTPUTS_EXPLAINATION].append(explanation_pre_outputs)
        self.final_outputs[CEVAELightning.FINAL_OUTPUTS_CRITIQUING].append(critiquing_pre_outputs)
    def on_fit_end(self):
        #save result
        result_path = os.path.abspath('../../lightning_logs/result.pkl')
        if (os.path.exists(result_path)):
            os.remove(result_path)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'wb') as f:
            pickle.dump(self.final_outputs, f, pickle.HIGHEST_PROTOCOL)

        #log
        ndcgs=map(lambda a: a['NDCG'], self.final_outputs[CEVAELightning.FINAL_OUTPUTS_ITEM])
        max_ndcg=max(ndcgs)
        self.mylog.log('final result:',
                       f'max_ndcg:{max_ndcg}',
                       'done!')

    def critiquing_validation(self, batch):
        result = dict()
        userid=batch[:,0].cpu().numpy()
        train_items_label = batch[:, 1:]
        rating, keyphrase, _, detached_latent, _, _ = self.model(train_items_label,sampling=False)
        #_,rating_indices=rating.topk(topk)
        explanation_rank_list = torch.argsort(keyphrase, dim=-1, descending=True)[:,:topk_keyphrase]
        explanation_rank_list = explanation_rank_list.cpu().numpy()
        affected_items_list = []
        for i, rank in enumerate(explanation_rank_list):
            key_choice = int(np.random.choice(rank, 1)[0])
            #indices = torch.tensor([key_choice])
            # Redistribute keyphrase prediction score
            rating_difference = keyphrase[i, key_choice]
            keyphrase[i, key_choice] = 0
            keyphrase_ratio = keyphrase[i] / torch.sum(keyphrase[i])
            keyphrase_redistribute_score = keyphrase_ratio * rating_difference
            keyphrase[i] += keyphrase_redistribute_score

            #affected_items = self.sp_keyphrase_item.index_select(0, indices).to_dense().view(-1).nonzero(as_tuple=True)[0].cpu().numpy()
            u=int(userid[i])
            # affected_items=self.train_df[self.train_df[USER_INDEX]==u
            #               & self.train_df[KEY_VECTOR].apply(lambda x:key_choice in x)][ITEM_INDEX].values
            #此处用dict加速
            affected_items=self.u_k_i_map[(u,key_choice)]
            affected_items_list.append(affected_items)
        modified_rating, modified_keyphrase = self.model.looping_predict(keyphrase, detached_latent)
        k_arr = [5, 10, 20]
        fmap_results = [[] for _ in k_arr]
        for i, k in enumerate(k_arr):
            for j, r in enumerate(rating):
                _,modified_r_indices = modified_rating[j].topk(k)
                affected_items = affected_items_list[j]
                _,r_indices = r.topk(k)
                r_indices=r_indices.cpu().numpy()
                modified_r_indices=modified_r_indices.cpu().numpy()
                fmap_results[i].append(average_precisionk(r_indices, np.isin(r_indices, affected_items))
                                       - average_precisionk(modified_r_indices, np.isin(modified_r_indices, affected_items)))
        fmap_results_dict = dict()
        for i, k in enumerate(k_arr):
            fmap_results_dict['F-MAP@{0}'.format(k)] = np.average(fmap_results[i])
        result['critiquing'] = fmap_results_dict
        return result

    def general_validation(self, batch):
        result = dict()
        train_items_label = batch[:, :self.item_num]
        train_keyphrases_label = batch[:, self.item_num:self.word_num]
        test_items_label = batch[:, self.item_num + self.word_num:self.item_num * 2 + self.word_num]
        test_keyphrases_label = batch[:, self.item_num * 2 + self.word_num:]
        rating, keyphrase, _, _, _, _ = self.model(train_items_label,sampling=False)
        # ------evaluate item recommendation --------
        atK = []
        tmpK = 5
        while tmpK <= topk:
            atK.append(tmpK)
            tmpK += 5
        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
        prediction = predict(rating, topk, train_items_label)
        item_eval = evaluate(prediction.cpu().numpy(), test_items_label.cpu().numpy(), metric_names, atK)
        result['item'] = item_eval
        # ------evaluate explaination recommendation --------
        atK = []
        tmpK = 5
        while tmpK <= topk_keyphrase:
            atK.append(tmpK)
            tmpK += 5
        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
        keyphrase_prediction = predict_keyphrase(keyphrase, topk_keyphrase)
        explanation_eval = evaluate(keyphrase_prediction.cpu().numpy(), test_keyphrases_label.cpu().numpy(), metric_names,
                                    atK)
        result['explanation'] = explanation_eval
        return result
    def total_eval(self, outputs,item_str):
        keys = outputs[0][item_str].keys()
        result = {key: np.average([o[item_str][key] for o in outputs]) for key in keys}
        return result



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,weight_decay=l2_weight)
        return optimizer

    def forward(self, input: Tensor, **kwargs) -> tuple:
        items = input[:, :self.item_num]

        return self.model(items,sampling=False)

    def loss_function(self, *, rating, rating_label, keyphrase, keyphrase_label, reconstructed_latent,
                          detached_latent,
                          mean, logvar):
        rating_loss = F.mse_loss(rating, rating_label)
        keyphrase_loss=F.mse_loss(keyphrase,keyphrase_label)
        recons_loss=F.mse_loss(reconstructed_latent,detached_latent)
        kl_loss = torch.mean(0.5 * torch.sum(torch.square(mean) + torch.exp(logvar) - logvar - 1, dim=1), dim=0)

        loss = rating_weight * rating_loss \
               + keyphrase_weight * keyphrase_loss \
               + recons_weight * recons_loss
        return loss,rating_loss,keyphrase_loss,recons_loss

    def looping_predict(self,*,modified_keyphrase,old_latent):
        rating,keyphrase=self.model.looping_predict(modified_keyphrase=modified_keyphrase,old_latent=old_latent)
        return rating,keyphrase



