import time
import matplotlib.pyplot as plt
import os
import numpy as np
import pytorch_lightning as pl
from data_model.beer_vae_data_module import BeerVaeDataModule
import os.path as path
import pandas as pd
from model.ce_vae.ce_vae_lightning import CEVAELightning
from pytorch_lightning.callbacks import ModelCheckpoint
from model.ce_vae.params import *
def main():


    data_module = BeerVaeDataModule(batch_size=batch_size)
    model_lightning = CEVAELightning(
        word_num=data_module.num_word,
        item_num=data_module.num_item)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc',save_top_k=1,mode='max')
    trainer = pl.Trainer(default_root_dir='../..',
                         gpus=1,
                         max_epochs=1000,
                         fast_dev_run=False,
                         # checkpoint_callback=True,
                         callbacks=[checkpoint_callback],
                         check_val_every_n_epoch=1,
                         num_sanity_val_steps=0)
    trainer.fit(model_lightning, data_module)

if __name__ == '__main__':
    main()



