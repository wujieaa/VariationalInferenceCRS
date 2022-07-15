import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch import Tensor
from model.ce_vae.params import *

class CEVAE(nn.Module):
    def __init__(self,
                 *,word_num,item_num) -> None:
        super().__init__()
        self.word_num = word_num
        self.item_num = item_num
        self.encode_layer=nn.Sequential(nn.Linear(item_num,latent_dim*2)
                                        )
        self.rating_prediction_layer=nn.Sequential(nn.Linear(latent_dim,item_num),
                                                   #nn.Sigmoid()
                                                   )
        self.keyphrase_prediction_layer=nn.Sequential(nn.Linear(latent_dim,word_num),
                                                      #nn.Sigmoid()
                                                      )
        self.reconstructed_latent_layer=nn.Sequential(nn.Linear(word_num,latent_dim),
                                                      nn.ReLU()
                                                      )

    def forward(self,x,*,sampling=True):
        mask1 = F.dropout(torch.ones_like(x))
        x=x*mask1
        encoded=self.encode_layer(x)
        mean=F.relu(encoded[:,:latent_dim])
        logvar=torch.tanh(encoded[:,latent_dim:])*3
        if sampling:
            stdvar = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(stdvar, dtype=torch.float, requires_grad=False)
            latent=mean+epsilon*stdvar
        else:
            latent=mean
        detached_latent=latent.detach()
        rating=self.rating_prediction_layer(latent)
        keyphrase=self.keyphrase_prediction_layer(latent)
        reconstructed_latent=self.reconstructed_latent_layer(keyphrase)
        return rating,keyphrase,reconstructed_latent,detached_latent,mean,logvar

    @torch.no_grad()
    def looping_predict(self,modified_keyphrase,old_latent):
        modified_latent=self.reconstructed_latent_layer(modified_keyphrase)
        latent=rou * modified_latent+(1-rou)*old_latent
        rating=self.rating_prediction_layer(latent)
        keyphrase=self.keyphrase_prediction_layer(latent)
        return rating,keyphrase



