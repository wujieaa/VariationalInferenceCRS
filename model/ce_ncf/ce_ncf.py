import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch import Tensor
from model.ce_ncf.params import *

class CENCF(nn.Module):
    def __init__(self,
                 *,word_num,user_num,item_num) -> None:
        super().__init__()
        self.word_num = word_num
        self.user_num = user_num
        self.item_num = item_num
        self.user_embeddings=nn.Embedding(user_num,embedding_dim)
        self.item_embeddings=nn.Embedding(item_num,embedding_dim)
        self.latent_layer=nn.Sequential(nn.Linear(embedding_dim*2,embedding_dim*2)
                                        ,nn.ReLU()
                                        )
        self.rating_prediction_layer=nn.Sequential(nn.Linear(embedding_dim*2,1),
                                                   #nn.Sigmoid()
                                                   )
        self.keyphrase_prediction_layer=nn.Sequential(nn.Linear(embedding_dim*2,word_num),
                                                      #nn.Sigmoid()
                                                      )
        self.reconstructed_latent_layer=nn.Sequential(nn.Linear(word_num,embedding_dim*2),
                                                      nn.ReLU())

    def forward(self,user,item):
        user_emb= self.user_embeddings(user)
        item_emb=self.item_embeddings(item)
        emb_cat=torch.cat((user_emb,item_emb),1)
        latent=self.latent_layer(emb_cat)
        detached_latent=latent.detach()
        rating=self.rating_prediction_layer(latent)
        keyphrase=self.keyphrase_prediction_layer(latent)
        reconstructed_latent=self.reconstructed_latent_layer(keyphrase)
        return rating,keyphrase,reconstructed_latent,detached_latent

    @torch.no_grad()
    def looping_predict(self,modified_keyphrase,old_latent):
        modified_latent=self.reconstructed_latent_layer(modified_keyphrase)
        latent=rou * modified_latent+(1-rou)*old_latent
        rating=self.rating_prediction_layer(latent)
        keyphrase=self.keyphrase_prediction_layer(latent)
        return rating,keyphrase



