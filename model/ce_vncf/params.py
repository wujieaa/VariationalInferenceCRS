learning_rate=1e-4
l2_weight=1e-5#l2范数
rating_weight=1.0
keyphrase_weight=1.0
recons_weight=1.0
klloss_weight=0.01
rou=0.5 #论文里混合原隐变量和更改后的隐变量的系数ρ
batch_size=1024 #16384
embedding_dim=10
topk=20 #商品推荐
topk_keyphrase=10 #关键字推荐
num_sampled_users=500#num_sampled_users_for_Critiquing_evaluation
