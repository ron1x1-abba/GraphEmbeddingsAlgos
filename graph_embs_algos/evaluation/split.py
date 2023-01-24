from typing import Union
import numpy as np
import torch


def train_test_split(X, test_size: Union[int, float] = 0.2, allow_duplicate: bool = False):
    """
    Generates test set which contains only those items that occured in train.
    :param X: torch.Tensor of shape (n, 3). The dataset to split.
    :param test_size: ratio of total triplets to be test.
    :param allow_duplicate: If test set can have duplicates.
    :return: X_train -- train set, X_test -- test set.
    """

    if isinstance(test_size, float):
        test_size = int(len(X) * test_size)

    X_train = None
    X_test_cands = X

    ents, ents_cnt = torch.unique(torch.cat([X_test_cands[:, 0], X_test_cands[:, 2]], dim=0), return_counts=True)
    rels, rels_cnt = torch.unique(X_test_cands[:, 1], return_counts=True)
    dict_ents = dict(zip(ents, ents_cnt))
    dict_rels = dict(zip(rels, rels_cnt))
    idx_train, idx_test = [] , []

    all_idx_perm = torch.randperm(X_test_cands.shape[0])
    for i, idx in enumerate(all_idx_perm):
        test_triple = X_test_cands[idx]
        dict_ents[test_triple[0]] -= 1
        dict_rels[test_triple[1]] -= 1
        dict_rels[test_triple[2]] -= 1

        if dict_ents[test_triple[0]] > 0 and dict_rels[test_triple[1]] > 0 and dict_ents[test_triple[2]] > 0:
            idx_test.append(idx)
            if len(idx_test) == test_size:
                idx_train.extend(all_idx_perm[i+1:].tolist())
                break
        else:
            dict_ents[test_triple[0]] += 1
            dict_rels[test_triple[1]] += 1
            dict_rels[test_triple[2]] += 1
            idx_train.append(idx)

    if len(idx_test) != test_size:
        if allow_duplicate:
            duplicate_idx = np.random.choice(idx_test, size=test_size - len(idx_test)).tolist()
            idx_test.extend(duplicate_idx)
        else:
            raise Exception("Cannot create test split due to not enough entities to occur in train and test."
                            "Set allow duplicate=True or reduce test size.")

    if X_train is None:
        X_train = X_test_cands[idx_train]
    X_test = X_test_cands[idx_test]

    return X_train[torch.randperm(X_train.shape[0])], X_test[torch.randperm(X_test.shape[0])]

