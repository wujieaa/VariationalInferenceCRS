import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader,  Dataset
import pandas as pd
import os.path as path
import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from data_model.data_const import *
import torch
from model.ce_vae.params import num_sampled_users
class BeerSparseDataModule(pl.LightningDataModule):

    def __init__(self, *, batch_size: int = 32):
        super().__init__()
        self.batch_size=batch_size
        train_item=BeerSparseDataModule.to_sparse_tensor(path.join(data_dir,'Rtrain.npz'))
        train_keyphrase=BeerSparseDataModule.to_sparse_tensor(path.join(data_dir,'Rkeyphrasetrain.npz'))

        test_item=BeerSparseDataModule.to_sparse_tensor(path.join(data_dir,'Rtest.npz'))
        test_keyphrase = BeerSparseDataModule.to_sparse_tensor(path.join(data_dir, 'Rkeyphrasetest.npz'))
        self.train_dataset= BeerTrainDataset(train_item,train_keyphrase)
        self.general_valid_dataset= BeerGeneralValidDataset(train_item, train_keyphrase, test_item, test_keyphrase)
        self.critiquing_valid_dataset=BeerCritiquingValidDataset(train_item,train_keyphrase,num_sampled_users,3)
        self.num_item=train_item.shape[1]
        self.num_word=train_keyphrase.shape[1]
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return [DataLoader(dataset=self.general_valid_dataset, batch_size=self.batch_size, shuffle=False),
                DataLoader(dataset=self.critiquing_valid_dataset, batch_size=self.batch_size, shuffle=False)
                ]

    def to_sparse_tensor( path):
        Acsr = sparse.load_npz(path)
        Acoo = Acsr.tocoo()
        data = torch.sparse.FloatTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                                        torch.FloatTensor(Acoo.data.astype(np.float)))
        return data

class BeerTrainDataset(Dataset):
    def __init__(self, train_item_data,train_keyphrase_data):
        self.item_data = train_item_data
        self.keyphrase_data = train_keyphrase_data
        self.len=self.item_data.shape[0]
    def __getitem__(self, index):
        indices=torch.tensor([index])
        items=self.item_data.index_select(0, indices).to_dense().view(-1)
        keyphrases=self.keyphrase_data.index_select(0, indices).to_dense().view(-1)
        return torch.cat((items,keyphrases),dim=0)
    def __len__(self):
        return self.len

class BeerGeneralValidDataset(Dataset):
    def __init__(self, train_item_data,train_keyphrase_data,test_item_data,test_keyphrase_data):
        self.train_item_data = train_item_data
        self.train_keyphrase_data = train_keyphrase_data
        self.test_item_data=test_item_data
        self.test_keyphrase_data=test_keyphrase_data
        self.len=self.train_item_data.shape[0]
    def __getitem__(self, index):
        indices=torch.tensor([index])
        train_item=self.train_item_data.index_select(0, indices).to_dense().view(-1)
        train_keyphrase=self.train_keyphrase_data.index_select(0, indices).to_dense().view(-1)
        test_item = self.test_item_data.index_select(0, indices).to_dense().view(-1)
        test_keyphrase = self.test_keyphrase_data.index_select(0, indices).to_dense().view(-1)
        return torch.cat((train_item,train_keyphrase,test_item,test_keyphrase),dim=0)
    def __len__(self):
        return self.len
class BeerCritiquingValidDataset(Dataset):
    def __init__(self, train_item_data, train_keyphrase_data,  num_users_sampled, repeat=3):
        self.train_item_data = train_item_data
        self.train_keyphrase_data = train_keyphrase_data
        self.len=num_users_sampled*repeat
        self.num_users_sampled=num_users_sampled
        self.userindices= np.random.choice(train_item_data.shape[0], num_users_sampled)
    def __getitem__(self, index):
        user_index=self.userindices[index % self.num_users_sampled]
        indices=torch.tensor([user_index],dtype=torch.long)
        train_item=self.train_item_data.index_select(0, indices).to_dense().view(-1)
        train_keyphrase=self.train_keyphrase_data.index_select(0, indices).to_dense().view(-1)
        return torch.cat((train_item,train_keyphrase),dim=0)
    def __len__(self):
        return self.len
