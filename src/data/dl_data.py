import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def dl_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data

# def dl_data_split(args, data):
#     X_train, X_valid, y_train, y_valid = train_test_split(
#                                                         data['train'].drop(['rating'], axis=1),
#                                                         data['train']['rating'],
#                                                         test_size=args.TEST_SIZE,
#                                                         random_state=args.SEED,
#                                                         shuffle=True
#                                                         )
#     data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
#     return data

def dl_data_split(args, data):
    count=data['train'].groupby("user_id").size()
    dfcount = pd.DataFrame(count, columns=["count"])
    data['train']=data['train'].merge(dfcount,on="user_id")
    data['train']=data['train'][1::]
    alrtrain = data['train'][data['train']["count"]!=1].drop(['count'],axis=1)
    newtrain1 = data['train'][data['train']["count"]==1].drop(['count'],axis=1)

    alr_train, alr_valid, alry_train, alry_valid = train_test_split(
                                                    alrtrain.drop(['rating'], axis=1),
                                                    alrtrain['rating'],
                                                    test_size = 0.11,
                                                    random_state=42, # args.SEED
                                                    shuffle=True
                                                    )


    data['X_train'], data['y_train'],  = alr_train, alry_train
    data['X_valid'] = pd.concat([newtrain1.drop(['rating'], axis=1),alr_valid])
    data['y_valid'] = pd.concat([newtrain1['rating'],alry_valid])
    return data


def dl_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
