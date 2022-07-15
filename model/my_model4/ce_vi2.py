import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from utils.model_utils import feature_fusion


class CEVI2(nn.Module):
    '''
    decode 加入user embeding
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
            nn.Linear(word_num+args.ui_embedding_dim, 1),
            #nn.Sigmoid()
        )

    def forward(self, user, item, is_training=True):
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)
        emb_cat=torch.cat([user_emb, item_emb], dim=1)
        z = self.encode_user_item(emb_cat)
        user_emb_z=torch.cat([user_emb,z],dim=1)
        rating_scores=self.rating(user_emb_z)
        return rating_scores, z

    @torch.no_grad()
    def looping_predict(self, keyphrase, critiqued_keyphrase_index,user=None):
        modified_keyhprase = feature_fusion(critiqued_keyphrase_index, keyphrase, self.args)
        user_emb=self.embedding_user(user)
        z=torch.cat([user_emb,modified_keyhprase],dim=1)
        rating_score = self.rating(z)
        return rating_score

