import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch import Tensor
from model.ce_vncf.params import *

class CEVNCFOrigin(nn.Module):
    def __init__(self,
                 *,word_num,user_num,item_num) -> None:
        super().__init__()
        self.word_num = word_num
        self.user_num = user_num
        self.item_num = item_num
        self.user_embeddings=nn.Embedding(user_num,embedding_dim)
        self.item_embeddings=nn.Embedding(item_num,embedding_dim)
        self.latent_layer=nn.Sequential(nn.Linear(embedding_dim*2,embedding_dim*4),
                                        )
        self.rating_prediction_layer=nn.Sequential(nn.Linear(embedding_dim*2,1),
                                                   #nn.Sigmoid()加上以后fmap为0
                                                   )
        self.keyphrase_prediction_layer=nn.Sequential(nn.Linear(embedding_dim*2,word_num),
                                                      #nn.Sigmoid()
                                                      )
        self.reconstructed_latent_layer=nn.Sequential(nn.Linear(word_num,embedding_dim*2),
                                                      nn.ReLU())

    def forward(self,user,item,*,sampling=True):
        user_emb= self.user_embeddings(user)
        item_emb=self.item_embeddings(item)
        emb_cat=torch.cat((user_emb,item_emb),1)
        #emb_cat=F.dropout(emb_cat)
        latent=self.latent_layer(emb_cat)
        mean=F.relu(latent[:,:embedding_dim*2])
        logvar=torch.tanh(latent[:,embedding_dim*2:])*3
        if sampling:
            stdvar = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(stdvar, dtype=torch.float, requires_grad=False)
            latent=mean+epsilon*stdvar
        else:
            latent=mean
        detached_latent=mean.detach()
        rating=self.rating_prediction_layer(latent)
        keyphrase=self.keyphrase_prediction_layer(latent)
        reconstructed_latent=self.reconstructed_latent_layer(keyphrase)
        return rating,keyphrase,reconstructed_latent,detached_latent,mean,logvar

    @torch.no_grad()
    def looping_predict(self,modified_keyphrase,old_latent):
        modified_latent=self.reconstructed_latent_layer(modified_keyphrase)
        #modified_latent=F.relu(modified_latent)
        latent=rou * modified_latent+(1-rou)*old_latent
        rating=self.rating_prediction_layer(latent)
        keyphrase=self.keyphrase_prediction_layer(latent)
        return rating,keyphrase



