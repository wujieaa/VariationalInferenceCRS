import unittest
from typing import Optional

import matplotlib.pyplot as plt

import os
import numpy as np
from torchvision.utils import make_grid
from torchvision.utils import save_image
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from utils.sampler import Negative_Sampler
from beer_data_module import *
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from functools import reduce
from utils.sampler import neg_sample

from data_model.data_const import *

class CatVaeCritiquingTest(unittest.TestCase):
    def setUp(self) -> None:
        pass
    def test_pd(self):
        df = pd.read_csv('d:/db/rec/beer/Train.csv')
        print(df[KEY_VECTOR].iloc[0])
    def test_dataset(self):
        ds=BeerDataset('d:/db/rec/beer/Valid.csv')
        t= ds.__getitem__(0)
        print(t)
        l=ds.__len__()
        print(l)
    def test_neg_sampler_dataset(self):
        data_dir='d:/db/rec/beer/'
        df = pd.read_csv(path.join(data_dir, 'UserIndex.csv'))
        user_num = df['UserIndex'].nunique()
        df = pd.read_csv(path.join(data_dir, 'ItemIndex.csv'))
        item_num = df['ItemIndex'].nunique()
        df = pd.read_csv(path.join(data_dir, 'KeyPhrases.csv'))
        word_num = len(df)
        test_dataset = BeerNegSamplerDataset(path.join(data_dir, 'Test.csv'),item_num, word_num)
        print(f'len:{test_dataset.__len__()}')
        data_loader=DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)
        i=1
        for data in data_loader:
            print(f'users:{data[0]}')
            print(f'items:{data[1]}')
            if i>3:
                break
            else:
                i=i+1

    def test_neg_sampler(self):
        # data_dir='d:/db/rec/beer/'
        # df_train = pd.read_csv('d:/db/rec/beer/Train.csv')
        # df_train.set_index(df_train.columns.values[0])
        # df = pd.read_csv(path.join(data_dir, 'KeyPhrases.csv'))
        # word_num = len(df)
        # df = pd.read_csv(path.join(data_dir, 'ItemIndex.csv'))
        # num_items=df['ItemIndex'].nunique()
        data={USER_INDEX:[0, 1, 2, 0, 1], ITEM_INDEX:[0, 1, 2, 3, 4], KEY_VECTOR:[[1, 2, 3], [1, 3, 5], [2, 4, 6], [0, 9], [5, 7]]}
        df_train=pd.DataFrame(data)
        word_num=10
        num_items=5
        negative_sampler = Negative_Sampler(df_train[[USER_INDEX,
                                                      ITEM_INDEX,
                                                      KEY_VECTOR]],
                                            USER_INDEX,
                                            ITEM_INDEX,
                                            RATING_ID,
                                            KEY_VECTOR,
                                            num_items,
                                            batch_size=3,
                                            num_keyphrases=word_num,
                                            negative_sampling_size=3)
        batches=negative_sampler.get_batches()
        print(batches)
    def test_list(self):
        ds=EmptyDataset()
        loader=DataLoader(dataset=ds,batch_size=1,shuffle=False)
        for data in loader:
            print(data)
    def test_csr_prepare(self):
        df = pd.read_csv('d:/program/db/rec/beer/Train.csv')
        usernum = df[USER_INDEX].nunique()
        itemnum = df[ITEM_INDEX].nunique()
        coli = []
        colj = []
        data = []
        for i in tqdm(range(usernum)):

            rated_items = df[df[USER_INDEX] == i][ITEM_INDEX].values
            for j in range(itemnum):
                if j not in rated_items:
                    coli.append(i)
                    colj.append(j)
                    data.append(1)
        df = pd.DataFrame({'coli': coli, 'colj': colj, 'data': data})
        df.to_csv(r'd:\tmp.csv')

    def test_csr(self):
        df=pd.read_csv(r'd:\tmp.csv')
        coli=df['coli'].values
        colj=df['colj'].values
        data=df['data'].values
        usernum=df['coli'].nunique()
        itemnum=df['colj'].nunique()
        m = csr_matrix((data, (coli, colj)), shape=(usernum, itemnum))
        arr=np.insert(m.getrow(1).toarray(),0,999)
        print(arr)
    def test_testdataset(self):
        df = pd.read_csv('d:/program/db/rec/beer/Train.csv')
        usernum=df[USER_INDEX].nunique()
        itemnum=df[ITEM_INDEX].nunique()

        ds=BeerNegTestDataset(df, usernum, itemnum, 75)
        loader=DataLoader(dataset=ds, batch_size=1, shuffle=False)
        i=0
        for data in loader:
            if(i<3):
                print(len(data))
                print(type(data[0]))
                print(len(data[0]))
                print('------------')
                i += 1
            else:
                break
    def test_csrmaxtrix(self):
        row = np.array([0, 1, 1, 0])
        col = np.array([0, 1, 2, 0])
        data = np.array([1, 2, 4, 8])
        m=csr_matrix((data, (row, col)), shape=(3, 3))
        print(m.toarray())
        t=torch.tensor(m.getrow(1).todense())
        col=torch.nonzero(t)[:,1]
        print(col)
    def test_csr_vstack(self):
        row = np.array([0, 1])
        col = np.array([0, 1])
        data = np.array([1, 2])
        m1=csr_matrix((data, (row, col)), shape=(3, 3))
        row = np.array([0, 2, 2])
        col = np.array([0, 2, 1])
        data = np.array([3, 4, 5])
        m2 = csr_matrix((data, (row, col)), shape=(3, 3))
        m=sp.vstack((m1,m2))
        print(m.toarray())

    def test_df(self):
        df=pd.DataFrame({'年份':[2000,2000,2001],'数学':[[1,2,3],[2],[3,4]]})
        df=df.groupby('年份')
        df=df.agg(lambda x:list(reduce(lambda a1,a2:set(a1)|set(a2),x)))
        df=df.reset_index()
        print(df)
        years=[]
        fens=[]
        for index,row in df.iterrows():
            year=row['年份']
            for fen in row['数学']:
                years.append(year)
                fens.append(fen)
        print(years)
        print(fens)

    def test_df2(self):
        df1 = pd.DataFrame({'年份': [2000, 2000,2000,2000,2000, 2001], 'all': [1,2,3,4,5,3]})
        df2 = pd.DataFrame({'年份': [2000,  2001], 'pos': [1,3]})
        df1=df1[['年份','all']].groupby('年份')['all'].apply(list)
        df1.rename('all',inplace=True)
        df2 = df2[['年份', 'pos']].groupby('年份')['pos'].apply(list).rename('pos')
        df=pd.concat([df1,df2],axis=1)
        df['neg']=df.apply(lambda x:neg_sample(np.arange(7),x[0],min(7-len(x[0]),len(x[1])*3)),axis=1)
        print(df)
    def test_df3(self):
        def neg_sample(all_items, all_pos_items, sampling_size):
            all_neg_items = np.setdiff1d(all_items, all_pos_items)
            return np.random.choice(all_neg_items, sampling_size, replace=False)
        num_items=10
        item_ids = np.arange(num_items)

        train_data_with_neg=pd.DataFrame({USER_INDEX:[1,2,3,4,5,6],
                                         ALL_POS_ITEM_LIST:[[3,4],[4],[5],[6],[7],[8]],
                                          TRAIN_NEG_ITEM_LIST:[[5],[5],[6],[7],[8],[1]]})

        df=train_data_with_neg.set_index(USER_INDEX)\
            .apply(lambda x:neg_sample(item_ids,x[ALL_POS_ITEM_LIST],3),
                   axis=1
                   )
        print(df)
        # print(df.iloc[1])
    def test_df4(self):
        df = pd.DataFrame({'year': [2000, 2000, 2001],'val':[1,2,3], 'vector': [[1,2,3],[1,2],[4,5,1]]})
        groupby=df.groupby('year')
        val=groupby['val'].apply(list)
        print(val)

        def sparse(x):
            v = np.zeros(6, dtype=np.int)
            for index in x:
                v[index] = 1
            return v.tolist()
        def compute_p_user(keys):
            count = np.zeros(6, dtype=np.int)
            for key in keys:
                v=np.array(sparse(key))
                count=count+v
            total=np.sum(count)
            p=count*1.0/total
            return p
        vector=groupby['vector'].apply(compute_p_user)
        cat=pd.concat([val,vector],axis=1)
        print(cat)
        # df=df.set_index('year')['vector'].apply(pd.Series).stack()
        # df=df.reset_index(level=1,drop=True).rename('item').astype('int').reset_index()
        # print(df)
        # df = pd.DataFrame({'year': [2000, 2001], 'vector': [[1, 2, 3], [1, 2]]})
        # def func(x):
        #     v=np.zeros(5,dtype=np.int)
        #     for index in x:
        #         v[index]=1
        #     return v
        # df[KEY_SPARSE] = df['vector'].apply(func)
        #
        # print(df)
    def test_np(self):
        p = np.array([1.0,0,0.5,0.6])
        min_val = 0.0001
        max_val = 0.9999
        p[p > max_val] = max_val
        p[p < min_val] = min_val
        print(p)

    def test_torch_argsort(self):
        a=torch.randn((3,4))
        print(a)
        sorted=torch.argsort(a,descending=True)
        print(sorted)
        print(a[:,sorted[:,:2]])







