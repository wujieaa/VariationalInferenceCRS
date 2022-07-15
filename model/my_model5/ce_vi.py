import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from utils.model_utils import reparameterize, feature_fusion


class CEVI(nn.Module):
    '''
    正态分布
    '''
    def __init__(self,
                 *, word_num, user_num, item_num, args) -> None:
        super().__init__()
        self.args = args
        self.word_num = word_num
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_user = nn.Embedding(user_num, args.ui_embedding_dim)
        self.embedding_item = nn.Embedding(item_num, args.ui_embedding_dim)
        self.encode_user_item = nn.Sequential(
            nn.Linear(args.ui_embedding_dim*2,args.encode_tensor_n),
            nn.ReLU(),
            nn.Linear(args.encode_tensor_n,word_num*2),

        )
        self.temp=1.0

        self.rating = nn.Sequential(
            nn.Linear(word_num, 1),
            #nn.Sigmoid()
        )

    def forward(self, user, item, is_training=True):
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)
        emb_cat=torch.cat([user_emb, item_emb], dim=1)

        latent = self.encode_user_item(emb_cat)
        mean = torch.sigmoid(latent[:, :self.word_num])
        logvar = torch.tanh(latent[:, self.word_num :]) * 3
        if self.training:
            stdvar = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(stdvar, dtype=torch.float, requires_grad=False)
            z = mean + epsilon * stdvar
        else:
            z = mean

        rating_scores=self.rating(z)

        # if self.is_nan(rating_scores):
        #     raise Exception('emb_cat is nan')
        return rating_scores, mean,logvar

    def is_nan(self,t):
        return torch.isnan(t).any() or torch.isinf(t).any()

    @torch.no_grad()
    def looping_predict(self, keyphrase, critiqued_keyphrase_index, user=None):
        modified_keyhprase = feature_fusion(critiqued_keyphrase_index, keyphrase, self.args)
        rating_score = self.rating(modified_keyhprase)
        return rating_score
