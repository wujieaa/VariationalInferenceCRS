import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from utils.model_utils import feature_fusion


class CEVI(nn.Module):
    '''
    伯努利分布隐变量
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
            nn.Linear(args.encode_tensor_n,word_num),
            nn.Sigmoid()
        )

        self.rating = nn.Sequential(
            # nn.Linear(word_num, word_num*2),
            # nn.ReLU(),
            nn.Linear(word_num, 1),
            #nn.Sigmoid()
        )

    def forward(self, user, item, is_training=True):
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)
        emb_cat=torch.cat([user_emb, item_emb], dim=1)
        z = self.encode_user_item(emb_cat)
        rating_scores=self.rating(z)
        return rating_scores, z

    @torch.no_grad()
    def looping_predict(self, keyphrase, critiqued_keyphrase_index,*params):
        modified_keyhprase = feature_fusion(critiqued_keyphrase_index, keyphrase, self.args)
        rating_score = self.rating(modified_keyhprase)
        return rating_score


