import torch
from torch import Tensor
def reparameterize( p: Tensor, temp) -> Tensor:
    '''伯努利分布的gumbel_softmax，别的分布不适用'''
    eps = 1e-20
    u1 = torch.rand_like(p)
    g1 = (torch.log(p) - torch.log(- torch.log(u1 + eps) + eps)) / temp
    u2 = torch.rand_like(p)
    g2 = (torch.log(1 - p) - torch.log(- torch.log(u2 + eps) + eps)) / temp
    g_cat = torch.cat([torch.unsqueeze(g1, dim=-1), torch.unsqueeze(g2, dim=-1)], dim=-1)
    g_max = torch.max(g_cat, dim=-1).values
    #
    exp1 = torch.exp(g1 - g_max)
    exp2 = torch.exp(g2 - g_max)
    res=exp1 / (exp1 + exp2)
    # if torch.isnan(res).any():
    #     if torch.isnan(p).any():
    #         print(p.tolist())
    #         raise Exception('p is nan at model utils')
    #
    #     raise Exception('is nan at model utils')
    return res
def feature_fusion(critiqued_keyphrase_index, keyphrase,args):
    modified_keyhprase = torch.clone(keyphrase)
    if args.fusion_type=='set_zero':
        # set  zeros
        modified_keyhprase[:, critiqued_keyphrase_index] = 0
    elif args.fusion_type=='zero_out':
        #zero-out
        min_val = torch.min(keyphrase, dim=1).values
        modified_keyhprase[:, critiqued_keyphrase_index] = min_val
        modified_keyhprase=args.rou*modified_keyhprase+(1-args.rou)*keyphrase
    return modified_keyhprase
