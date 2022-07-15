import os
import pickle

import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import os.path as path
import torch
import numpy as np
from ast import literal_eval
from utils.sampler import Negative_Sampler,neg_sample
from tqdm import tqdm
from scipy.sparse import csr_matrix
from data_model.data_const import *

class BeerDataModule(pl.LightningDataModule):

    def __init__(self, *,train_df,
                 test_df,all_df,user_num,item_num,word_num,args):
        super().__init__()
        self.train_df=train_df
        self.test_df=test_df
        self.all_df=all_df
        self.user_num=user_num
        self.item_num=item_num
        self.word_num=word_num
        self.batch_size=args.batch_size
        self.train_data_file_name= 'beer_train_data.npy'
        self.user_grouped_data_file_name = 'beer_user_group_df.pkl'
        self.args=args

    def prepare_data(self, *args, **kwargs):
        train_data_path=path.join(data_dir, self.train_data_file_name)
        user_grouped_data_path=path.join(data_dir, self.user_grouped_data_file_name)
        if path.exists(train_data_path):
            print('loading datas from the disk')
            user_grouped_df=pd.read_pickle(user_grouped_data_path)
            train_data=np.load(train_data_path)
        else:
            print('-------------gen train data------------------')
            user_grouped_df = self.gen_data()
            train_data = self.gen_train_data(user_grouped_df)
            # 内存溢出
            # print('-------------gen valid data---------------------')
            # test_data = self.gen_valid_data(df)
            print('saving')
            user_grouped_df.to_pickle(user_grouped_data_path)
            np.save(train_data_path,train_data)
        self.train_dataset = SimpleDataset(train_data)
        self.test_dataset=ValidDataset(self.test_df,user_grouped_df[LEFT_NEG_ITEM_LIST],self.word_num,self.args)
    def setup(self, stage=None):
        pass
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)
    # def test_dataloader(self):
    #     return DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)
    def sparse(self,x):
        v = np.zeros(self.word_num, dtype=np.int)
        for index in x:
            v[index] = 1
        return v.tolist()
    def gen_data(self):
        print('grouping datas by user index')
        train_series = self.train_df[[USER_INDEX, ITEM_INDEX]] \
            .groupby(USER_INDEX)[ITEM_INDEX].apply(list).rename(TRAIN_POS_ITEM_LIST)
        # test_series = self.test_df[[USER_INDEX, ITEM_INDEX]] \
        #     .groupby(USER_INDEX)[ITEM_INDEX].apply(list).rename(TEST_POS_ITEM_LIST)
        all_series = self.all_df[[USER_INDEX, ITEM_INDEX]] \
            .groupby(USER_INDEX)[ITEM_INDEX].apply(list).rename(ALL_POS_ITEM_LIST)
        print('concating datas')
        user_items_df = pd.concat([all_series, train_series], axis=1)
        all_items = np.arange(self.item_num)
        #all_pos_index, train_pos_index, test_pos_index, train_neg_index = 0, 1, 2, 3
        print('----------------gen TRAIN_NEG_ITEM_LIST---------------')
        user_items_df[TRAIN_NEG_ITEM_LIST] = \
            user_items_df.apply(lambda x: neg_sample(all_items,
                                          x[ALL_POS_ITEM_LIST],
                                          min(self.item_num - len(x[ALL_POS_ITEM_LIST]),
                                              self.args.neg_sampling_size * len(x[TRAIN_POS_ITEM_LIST]))),
                     axis=1)
        print('-------------gen LEFT_NEG_ITEM_LIST------------------')
        user_items_df[LEFT_NEG_ITEM_LIST] = \
            user_items_df.apply(lambda x: np.setdiff1d(all_items,
                                            np.concatenate([x[ALL_POS_ITEM_LIST], x[TRAIN_NEG_ITEM_LIST]])).tolist(),
                     axis=1)

        return  user_items_df

    def gen_valid_data(self, df):
        pos_test_df = self.test_df[[USER_INDEX, ITEM_INDEX, KEY_VECTOR]]
        pos_test_df.loc[:, RATING_ID] = 1
        # neg_test_df=df[[USER_INDEX,LEFT_NEG_ITEM_LIST]]
        print('stack neg items')
        neg_test_df = df[LEFT_NEG_ITEM_LIST] \
            .apply(pd.Series).stack().reset_index(level=1, drop=True)\
            .rename(ITEM_INDEX).astype('int').reset_index()
        neg_test_df[KEY_VECTOR] = pd.Series(data=[[] for _ in range(len(neg_test_df))])
        neg_test_df.loc[:, RATING_ID] = 0
        print('concat pos and neg datas')
        test_df = pd.concat([pos_test_df, neg_test_df])
        print('sparse KEY_VECTOR')
        test_df[KEY_SPARSE] = test_df[KEY_VECTOR].apply(self.sparse)
        test_data = test_df[[USER_INDEX, ITEM_INDEX, RATING_ID]].values
        print('KEY_SPARSE to Series')
        test_vec = test_df[KEY_SPARSE].apply(pd.Series).values
        print('concat')
        test_data = np.concatenate([test_data, test_vec], axis=1)
        return test_data

    def gen_train_data(self, df):
        pos_train_df = self.train_df[[USER_INDEX, ITEM_INDEX, KEY_VECTOR]]
        pos_train_df.loc[:, RATING_ID] = 1
        print('stack neg items')
        neg_train_df = df[TRAIN_NEG_ITEM_LIST] \
            .apply(pd.Series).stack().reset_index(level=1, drop=True).rename(ITEM_INDEX).astype('int').reset_index()
        neg_train_df[KEY_VECTOR] = pd.Series(data=[[] for _ in range(len(neg_train_df))])
        neg_train_df.loc[:, RATING_ID] = 0
        print('concat pos and neg datas')
        train_df = pd.concat([pos_train_df, neg_train_df])
        print('sparse KEY_VECTOR')
        train_df[KEY_SPARSE] = train_df[KEY_VECTOR].apply(self.sparse)
        train_data = train_df[[USER_INDEX, ITEM_INDEX, RATING_ID]].values
        print('KEY_SPARSE to Series')
        train_vec = train_df[KEY_SPARSE].apply(pd.Series).values
        print('concat')
        train_data = np.concatenate([train_data, train_vec], axis=1)
        return train_data



class EmptyDataset(Dataset):
    def __getitem__(self, index):
        return 1
    def __len__(self):
        return 1

class SimpleDataset(Dataset):
    def __init__(self,data):
        self.data=torch.tensor(data,dtype=torch.long)
        self.len=self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len
class ValidDataset(Dataset):
    def __init__(self, pos_test_df, neg_grouped_s, word_num, args):
        self.pos_test_df=pos_test_df
        self.neg_grouped_df=neg_grouped_s.reset_index()
        # self.len=neg_grouped_s.shape[0]
        self.len=args.valid_sampling_num
        self.word_num=word_num
        self.reset_user_indexes()
        self.args=args
    def reset_user_indexes(self):
        self.user_indexes=np.random.choice(self.neg_grouped_df.shape[0],self.len,replace=False)

    def __getitem__(self, index):
        if index == 0:
            self.reset_user_indexes()
            print('reset valid dataset user indexes')
        user_index=self.user_indexes[index]
        user_s= self.neg_grouped_df.iloc[user_index, :]
        return self.gen_single_data(user_s)
    def __len__(self):
        return self.len
    def sparse(self,x):
        v = np.zeros(self.word_num, dtype=np.int)
        for index in x:
            v[index] = 1
        return v.tolist()
    def gen_single_data(self, user_s):
        user_index=user_s[USER_INDEX]
        user_neg_items=user_s[LEFT_NEG_ITEM_LIST]

        pos_user_df = self.pos_test_df[self.pos_test_df[USER_INDEX] == user_index]
        user_pos_items = pos_user_df[ITEM_INDEX].values
        user_pos_ratings = np.ones_like(user_pos_items, dtype=np.int)
        user_pos_keyphrases = pos_user_df[KEY_VECTOR].apply(self.sparse).values.tolist()
        user_pos_keyphrases = np.array(user_pos_keyphrases, dtype=np.int)


        user_neg_items = np.array(user_neg_items, dtype=np.int)
        user_neg_items=np.random.choice(user_neg_items,self.args.valid_item_num_per_user-user_pos_items.shape[0],replace=False)
        user_neg_ratings=np.zeros_like(user_neg_items,dtype=np.int)
        user_neg_keyphrases=np.zeros((user_neg_ratings.shape[0],self.word_num),dtype=np.int)

        user_neg_items = user_neg_items.reshape(-1, 1)
        user_neg_ratings = user_neg_ratings.reshape(-1, 1)
        user_pos_items = user_pos_items.reshape(-1, 1)
        user_pos_ratings = user_pos_ratings.reshape(-1, 1)
        test_neg_data = np.concatenate([user_neg_items, user_neg_ratings,user_neg_keyphrases], axis=1)
        test_pos_data=np.concatenate([user_pos_items,user_pos_ratings,user_pos_keyphrases],axis=1)
        test_data=np.concatenate([test_pos_data,test_neg_data],axis=0)
        test_data=np.insert(test_data,0,user_index,axis=1)
        np.random.shuffle(test_data)
        return test_data