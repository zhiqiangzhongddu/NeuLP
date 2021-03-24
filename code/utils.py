import numpy as np
import pandas as pd
import scipy.sparse as sp
from copy import deepcopy
import random

import torch
from torch.utils.data import Dataset, DataLoader

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sample_negative(G_user, ratings, test_samples, num_negatives):
    """return all negative items & 100 sampled negative items"""
    user_pool = set(G_user.nodes())
    interact_status = ratings.groupby('u1')['u2'].apply(set).reset_index().rename(
        columns={'u2': 'interacted_u2'})

    self_friend = pd.DataFrame({'u1': list(G_user.nodes()),
                                'u2': list(G_user.nodes()),
                                'label': 1})
    test_samples = pd.concat([test_samples, self_friend]).sort_values(['u1', 'u2'],
                                                                      ascending=[True, True]). \
        reset_index(drop=True)

    _test_samples = deepcopy(test_samples)
    _test_samples = _test_samples.rename(columns={'u1': 'u2', 'u2': 'u1'})
    test_samples = pd.concat([test_samples, _test_samples]).sort_values(['u1', 'u2']).reset_index(drop=True)
    interact_status_test = test_samples.groupby('u1')['u2'].apply(set).reset_index().rename(
        columns={'u2': 'interacted_u2_test'})

    interact_status = pd.merge(interact_status, interact_status_test, on='u1', how='inner')

    interact_status['negative_u2'] = interact_status['interacted_u2'].apply(lambda x: user_pool - x)
    interact_status['negative_u2'] = interact_status['negative_u2'] - interact_status['interacted_u2_test']
    interact_status['negative_u2'] = interact_status['negative_u2'].apply(lambda x: random.sample(x, num_negatives))
    return interact_status[['u1', 'negative_u2']]


class UserUserRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_left_tensor, user_right_tensor, target_tensor):
        self.user_left_tensor = user_left_tensor
        self.user_right_tensor = user_right_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_left_tensor[index], self.user_right_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_left_tensor.size(0)


def instance_a_train_loader(train_ratings, negatives, num_negatives, batch_size):
    """instance train loader for one training epoch"""
    u1s, u2s, labels = [], [], []
    train_ratings = pd.merge(train_ratings, negatives[['u1', 'negative_u2']], on='u1')
    train_ratings['negatives'] = train_ratings['negative_u2'].apply(lambda x: random.sample(x, 1))
    for row in train_ratings.itertuples():
        u1s.append(int(row.u1))
        u2s.append(int(row.u2))
        labels.append(float(row.label))
        for i in range(num_negatives):
            u1s.append(int(row.u1))
            u2s.append(int(row.negatives[i]))
            labels.append(float(0))  # negative samples get 0 rating
    dataset = UserUserRatingDataset(user_left_tensor=torch.LongTensor(u1s),
                                    user_right_tensor=torch.LongTensor(u2s),
                                    target_tensor=torch.FloatTensor(labels))
    #                                     target_tensor=torch.LongTensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)