import unittest
import time

import matplotlib.pyplot as plt
import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from data_model.beer_data_module import BeerDataModule
import torch.nn as nn
import torch
from model.ce_ncf.params import *
from params import *
import os.path as path
import pandas as pd
from model.ce_ncf.ce_ncf_lightning import CENCFLightning
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.mylog import LogToFile
from ast import literal_eval
from data_model.data_const import *
class CeNcfTest(unittest.TestCase):
    def setUp(self) -> None:
        pass
    def test_path(self):
        name=path.join('d:/a','a.txt')
        print(name)
    def test_train(self):
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
                                     batch_size=batch_size)
        model_lightning = CENCFLightning(word_num=word_num,
                                         user_num=user_num,
                                         item_num=item_num)

        checkpoint_callback = ModelCheckpoint(monitor='val_acc')
        trainer = pl.Trainer(default_root_dir='../..',
                             gpus=1,
                             max_epochs=1,
                             fast_dev_run=False,
                             #checkpoint_callback=True,
                             callbacks=[checkpoint_callback],
                             check_val_every_n_epoch=1,
                             num_sanity_val_steps=0,
                             limit_train_batches=10,
                             limit_val_batches=10,
                             limit_test_batches=10
                             )
        trainer.fit(model_lightning, data_module)
    def test_file(self):
        self.mylog = LogToFile(os.path.join(os.path.abspath('../../lightning_logs'), 'log.txt'))
        self.mylog.clear()
        self.mylog.log('hello',
                       'world')
    def test_np(self):

        a=np.random.choice(10,3)
        print(a)
        print(np.average(a))
    def test_torch(self):
        a=torch.tensor([[1,1,0],[0,0,1]],dtype=torch.int64)
        index=torch.where(a==1)
        print(index)

    def test_lightning(self):
        data_module = myModule()
        model_lightning = Mylightning()

        trainer = pl.Trainer()
        trainer.fit(model_lightning, data_module)
class Mylightning(pl.LightningModule):
    def __init__(self):
        pass
    def training_step(self, x):
        print(x)
        y=torch.sum(x*2)

        return torch.nn.functional.linear(y,1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
class myModule(pl.LightningDataModule):
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(TestDataset(),batch_size=1,shuffle=False)
class TestDataset(Dataset):
    def __init__(self):
        self.len=2
        self.count=0
    def __getitem__(self, index):
        if self.count==0:
            return [[1,2],[1]]
        else:
            return [[3,4]]
    def __len__(self):
        return self.len





