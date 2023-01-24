import numpy as np
import torch


def hits_at_n_score(ranks, n):
    """
    The function computes how many elements of a vector of rankings ``ranks`` make it to the top ``n`` positions.

    :param ranks: torch.Tensor of shape (n,) or (n, 2)
    :param n: int
    :return: torch.FloatTensor
    """

    ranks = ranks.view(-1,)
    return (ranks <= n).sum() / len(ranks)


def mrr_score(ranks):
    """
    The function computes the mean of the reciprocal of elements of a vector of rankings ``ranks``.

    :param ranks: torch.Tensor of shape (n,) or (n,2)
    :return: torch.FloatTensor
    """

    ranks = ranks.view(-1,)
    return (1 / ranks).sum() / len(ranks)


def rank_score(y_true, y_pred, pos_label=1):
    """
    The rank of a positive element against a list of negatives.

    :param y_true: torch.Tensor of shape (n,) of binary labels. Tensor contains only 1 positive.
    :param y_pred: torch.Tensor of shape (n,) of score for 1 positive and n-1 negatives elems.
    :param pos_label: the value of positive label.
    :return: int : rank of positive element.
    """

    idx = torch.argsort(y_pred, descending=True)
    y_ord = y_true[idx]
    rank = (y_ord == pos_label).argmax()
    return rank


def mr_score(ranks):
    """
    The function computes the mean of a vector of rankings ``ranks``.

    :param ranks: torch.Tensor of shape (n,) of ranks
    :return:
    """

    ranks = ranks.view(-1,)
    return ranks.sum() / len(ranks)
