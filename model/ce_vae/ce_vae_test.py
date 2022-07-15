import unittest
import time

import matplotlib.pyplot as plt
import os
import numpy as np
import pytorch_lightning as pl
from data_model.beer_vae_data_module import BeerVaeDataModule
import torch.nn as nn
import torch
from model.ce_vae.params import *
from params import *
import os.path as path
import pandas as pd
from model.ce_vae.ce_vae_lightning import CEVAELightning
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.mylog import LogToFile
from data_model.data_const import data_dir
import scipy.sparse as sparse
class CeVaeTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_train(self):
        data_module = BeerVaeDataModule(batch_size=batch_size)
        model_lightning = CEVAELightning(
                                         word_num=data_module.num_word,
                                         item_num=data_module.num_item)

        checkpoint_callback = ModelCheckpoint(monitor='val_acc',save_top_k=1,mode='max')
        trainer = pl.Trainer(default_root_dir='../..',
                             gpus=1,
                             max_epochs=1,
                             fast_dev_run=False,
                             #checkpoint_callback=True,
                             callbacks=[checkpoint_callback],
                             check_val_every_n_epoch=1,
                             num_sanity_val_steps=0,
                             limit_train_batches=2,
                             limit_val_batches=2,
                             limit_test_batches=2
                             )
        trainer.fit(model_lightning, data_module)
    def test_file(self):
        self.mylog = LogToFile(os.path.join(os.path.abspath('../../lightning_logs'), 'log.txt'))
        self.mylog.clear()
        self.mylog.log('hello',
                       'world')
    def test_topk(self):
        t1=torch.arange(0,9)
        index=torch.randperm(9)
        t1=t1[index].view(3,3)
        print(t1)
        t2=t1.topk(2,dim=-1).indices
        print(t2)
    def test_tmp(self):
        df=pd.DataFrame({'a':[1,1,3],'c':['a','b','c'],'b':[[1,4,3],[4,5,6],[2,3,4]]})
        affected_items = df[df['a'] == 1 & df['b'].apply(lambda x: 4 in x)]['c'].values
        print(len(affected_items))
        data=df.values
        print(data)





