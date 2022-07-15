import unittest
import time

import matplotlib.pyplot as plt
import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from data_model.beer_data_module4 import BeerDataModule
import torch.nn as nn
import torch
import os.path as path
import pandas as pd
from model.my_model4.ce_vi_lightning import CEVILightning
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.mylog import LogToFile
from ast import literal_eval
from data_model.data_const import *
import argparse
class CeBvaeTest(unittest.TestCase):
    def setUp(self) -> None:
        pass
    def test_path(self):
        name=path.join('d:/a','a.txt')
        print(name)
    def test_train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--result_dir', default='../..')
        parser.add_argument('--result_file_name', default='cevi_result')
        parser.add_argument('--epoch', type=int, default=200)#测试不会循环200次
        parser.add_argument('--model_name', default='CEVI6')
        parser.add_argument('--fusion_type', default='set_zero')
        parser.add_argument('--encode_tensor_n', type=int, default=200)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--l2_weight', type=float, default=1e-5)
        parser.add_argument('--rating_weight', type=float, default=1.0)
        parser.add_argument('--keyphrase_weight', type=float, default=0.1)
        # parser.add_argument('--recon_weight', type=float,default=1.0)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--ui_embedding_dim', type=int, default=10)
        # parser.add_argument('--cluster_num', type=int,default=20)
        # parser.add_argument('--latent_dim', type=int,default=75)
        parser.add_argument('--topk', type=int, default=20)
        parser.add_argument('--topk_keyphrase', type=int, default=10)
        parser.add_argument('--valid_sampling_num', type=int, default=1000)
        parser.add_argument('--valid_item_num_per_user', type=int, default=1024)
        parser.add_argument('--neg_sampling_size', type=int, default=5)
        parser.add_argument('--rou', type=float, default=0.5)

        args = parser.parse_args()
        print('------args-----------')
        for k in list(vars(args).keys()):
            print(f'{k}:{vars(args)[k]}')
        print('------args-----------')
        df = pd.read_csv(path.join(data_dir, 'UserIndex.csv'))
        user_num = df['UserIndex'].nunique()
        df = pd.read_csv(path.join(data_dir, 'ItemIndex.csv'))
        item_num = df['ItemIndex'].nunique()
        keyphrase_df = pd.read_csv(path.join(data_dir, 'KeyPhrases.csv'))
        word_num = len(keyphrase_df)
        train_df = pd.read_csv(path.join(data_dir, 'Train.csv'))
        train_df = train_df.set_index(train_df.columns.values[0])
        train_df[KEY_VECTOR] = train_df[KEY_VECTOR].apply(literal_eval)
        # valid_df = pd.read_csv(path.join(data_dir, 'Valid.csv'))
        # valid_df.set_index(valid_df.columns.values[0])
        test_df = pd.read_csv(path.join(data_dir, 'Test.csv'))
        test_df = test_df.set_index(test_df.columns.values[0])
        test_df[KEY_VECTOR] = test_df[KEY_VECTOR].apply(literal_eval)

        all_df = pd.read_csv(path.join(data_dir, 'data.csv'))
        all_df = all_df.set_index(all_df.columns.values[0])
        all_df[KEY_VECTOR] = all_df[KEY_VECTOR].apply(literal_eval)

        data_module = BeerDataModule(train_df=train_df,
                                     test_df=test_df,
                                     all_df=all_df,
                                     user_num=user_num,
                                     item_num=item_num,
                                     word_num=word_num,
                                     args=args)
        model_lightning = CEVILightning(word_num=word_num,
                                        user_num=user_num,
                                        item_num=item_num,
                                        args=args)

        checkpoint_callback = ModelCheckpoint(monitor='val_acc')
        trainer = pl.Trainer(default_root_dir='../..',
                             gpus=1,
                             max_epochs=1,
                             fast_dev_run=False,
                             checkpoint_callback=True,
                             callbacks=[checkpoint_callback],
                             check_val_every_n_epoch=1,
                             num_sanity_val_steps=0,
                             limit_train_batches=10,
                             limit_val_batches=10,
                             limit_test_batches=10
                             )
        trainer.fit(model_lightning, data_module)




