import time
import matplotlib.pyplot as plt
import os
import numpy as np
import pytorch_lightning as pl
from data_model.beer_data_module import BeerDataModule
from model.ce_ncf.params import *
import os.path as path
import pandas as pd
from model.ce_ncf.ce_ncf_lightning import CENCFLightning
from pytorch_lightning.callbacks import ModelCheckpoint
from ast import literal_eval
from data_model.data_const import data_dir
from data_model.data_const import *
def main():
    df = pd.read_csv(path.join(data_dir, 'UserIndex.csv'))
    user_num = df['UserIndex'].nunique()
    df = pd.read_csv(path.join(data_dir, 'ItemIndex.csv'))
    item_num = df['ItemIndex'].nunique()
    keyphrase_df = pd.read_csv(path.join(data_dir, 'KeyPhrases.csv'))
    word_num = len(keyphrase_df)
    train_df = pd.read_csv(path.join(data_dir, 'Train.csv'))
    train_df=train_df.set_index(train_df.columns.values[0])
    train_df[KEY_VECTOR]=train_df[KEY_VECTOR].apply(literal_eval)
    # valid_df = pd.read_csv(path.join(data_dir, 'Valid.csv'))
    # valid_df.set_index(valid_df.columns.values[0])
    test_df = pd.read_csv(path.join(data_dir, 'Test.csv'))
    test_df=test_df.set_index(test_df.columns.values[0])
    test_df[KEY_VECTOR]=test_df[KEY_VECTOR].apply(literal_eval)

    all_df = pd.read_csv(path.join(data_dir, 'data.csv'))
    all_df=all_df.set_index(all_df.columns.values[0])
    all_df[KEY_VECTOR]=all_df[KEY_VECTOR].apply(literal_eval)

    #train_df = pd.concat([train_df, valid_df], ignore_index=True)

    data_module = BeerDataModule(train_df=train_df,
                                 test_df=test_df,
                                 all_df=all_df,
                                 user_num=user_num,
                                 item_num=item_num,
                                 word_num=word_num,
                                 batch_size=batch_size)
    model_lightning = CENCFLightning(
                                     word_num=word_num,
                                     user_num=user_num,
                                     item_num=item_num)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc')
    trainer = pl.Trainer(default_root_dir='../..',
                         gpus=1,
                         max_epochs=200,
                         fast_dev_run=False,
                         # checkpoint_callback=True,
                         callbacks=[checkpoint_callback],
                         check_val_every_n_epoch=1,
                         num_sanity_val_steps=0)
    trainer.fit(model_lightning, data_module)

if __name__ == '__main__':
    main()



