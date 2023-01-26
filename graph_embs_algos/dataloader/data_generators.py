import torch
from torch.utils.data import Dataset
import numpy as np

from typing import Union, List

class TripletDataset(Dataset):
    def __init__(self,
                 subjects: Union[List, np.ndarray, torch.Tensor],
                 predicates: Union[List, np.ndarray, torch.Tensor],
                 objects: Union[List, np.ndarray, torch.Tensor]
                 ):
        """
        :param subjects: indexes of subjects
        :param predicates: indexes of predicates
        :param objects: indexes of objects
        Contains data in form of matrix : (n_ex, 3) , where 3 goes to (subject, predicate, object)
        """
        if len(subjects) != len(predicates) or len(subjects) != len(objects):
            raise RuntimeError("Mismatch in triplets length!")

        if isinstance(subjects, list):
            subjects = torch.Tensor(subjects)
        elif isinstance(subjects, np.ndarray):
            subjects = torch.from_numpy(subjects)
        elif isinstance(subjects, torch.Tensor):
            pass
        else:
            raise TypeError("Subjects must be on of [list, np.ndarray, torch.Tensor]")

        if isinstance(objects, list):
            objects = torch.Tensor(objects)
        elif isinstance(objects, np.ndarray):
            objects = torch.from_numpy(objects)
        elif isinstance(objects, torch.Tensor):
            pass
        else:
            raise TypeError("Subjects must be on of [list, np.ndarray, torch.Tensor]")

        if isinstance(predicates, list):
            predicates = torch.Tensor(predicates)
        elif isinstance(predicates, np.ndarray):
            predicates = torch.from_numpy(predicates)
        elif isinstance(predicates, torch.Tensor):
            pass
        else:
            raise TypeError("Subjects must be on of [list, np.ndarray, torch.Tensor]")

        self.data = torch.cat([subjects.view(-1, 1), predicates.view(-1, 1), objects.view(-1, 1)], dim=1)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def generate_corruption_fit(
        triplets: torch.Tensor,
        entities_list: List[str] = None,
        eta: int = 1,
        corrupt: str = 's,o',
        ent_size: int = 0
    ) -> torch.Tensor:
    """
    Generate corrupted triplets for each positive triplet given.

    :param triplets: Tensor of shape (n, 3) of positive triplets. Will be used to generate negatives.
    :param entities_list: List of entities to use for corruption.
    :param eta: Number of negatives per 1 triplet.
    :param corrupt: Which part of triplet to corrupt. Can be 's' for subject, 'o' for object, 's+o' for both.
    :param ent_size: Size of entities to use for generating corrupted triplets.
    :return: torch.Tensor shape [n * eta, 3] of eta per positive corrupted triplets.
        For each raw in X corruption idxes can be found by [idx + i*n for i in range(eta)]

    If :param entities_list is None and ent_size=0, then batch entities will be used to generate corruptions.
    """
    if corrupt not in ['s', 'o', 's+o', 's,o']:
        raise ValueError(f"Invalid argument value {corrupt} passed for corruption type!")

    if corrupt == 's,o':
        corrupt = 's+o'

    if entities_list is not None:
        if isinstance(entities_list, np.ndarray):
            entities_list = torch.from_numpy(entities_list)
        elif isinstance(entities_list, list):
            entities_list = torch.Tensor(entities_list)
        elif isinstance(entities_list, torch.Tensor):
            pass
        else:
            raise TypeError("Entities list if implemented for [list, np.ndarray, torch.Tensor]. Please, use them!")

    dataset = torch.tile(triplets.view(-1,), (eta,)).view(-1, 3)

    if corrupt == 's+o':
        subj_mask = torch.randint(0, 2, (triplets.shape[0] * eta,)).bool() # in [0, 1]
    else:
        subj_mask = torch.ones(triplets.shape[0] * eta, dtype=torch.bool)
        if corrupt == 's':
            subj_mask = ~subj_mask

    obj_mask = (~subj_mask).int()
    subj_mask = subj_mask.int()

    if ent_size != 0:
        replacements = torch.randint(0, ent_size, (dataset.shape[0],))
    else:
        if entities_list is None:
            # use entities in batch
            entities_list = torch.unique(
                torch.cat([triplets[:, 0], triplets[:, 2]], dim=0)
            )

        rand_indices = torch.randint(0, entities_list.shape[0], (dataset.shape[0],))
        replacements = torch.gather(entities_list, 0, rand_indices)

    subjects = dataset[:, 0] * subj_mask + obj_mask * replacements
    relations = dataset[:, 1]
    objects = dataset[:, 2] * obj_mask + subj_mask * replacements
    return dataset, torch.cat([subjects.view(-1, 1), relations.view(-1, 1), objects.view(-1, 1)], dim=1)


def generate_corruption_eval(
        triplets: torch.Tensor,
        entities_for_corrupt: torch.Tensor,
        corrupt: str = 's,o',
        use_filter: bool = False,
        pos_filter: dict = None
    ) -> torch.Tensor:
    """
    Generate corruptions for evaluation.
    :param triplets: torch.Tensor of shape (1, 3) of positive triplets. Will be used to generate negatives.
    :param entities_for_corrupt: Entities IDs which will be used to generate corruptions.
    :param corrupt: Which part of triplet to corrupt. Can be 's' for subject, 'o' for object, 's+o' for both.
    :param use_filter: Whether to filter FN in corrupted triplets.
    :param pos_filter: Dict of all positive triplets {(s_idx, p_idx, o_idx) : True}
    :return: torch.Tensor os shape (n, 3) of corrupted triplets. Where n -- len entities for corrupt.
    """
    if use_filter and pos_filter is None:
        raise RuntimeError("Filter must be set when parameter use_filter set True.")
    if corrupt not in ['s', 'o', 's+o', 's,o']:
        raise ValueError(f"Invalid argument value {corrupt} passed for corruption type!")

    if corrupt == 's,o':
        corrupt = 's+o'

    if corrupt in ['s+o', 'o']:  # object is corrupted, leave subjects as it is
        rep_subj = triplets[:, 0].view(-1, 1).repeat((1, entities_for_corrupt.shape[0])) # shape (n, len(ent_for_corr))

    if corrupt in ['s+o', 's']:  # subject is corrupted, leave objects as it is
        rep_obj = triplets[:, 2].view(-1, 1).repeat((1, entities_for_corrupt.shape[0]))

    rep_rel = triplets[:, 1].view(-1, 1).repeat(1, entities_for_corrupt.shape[0])

    rep_ent = entities_for_corrupt.repeat(triplets.shape[0], 1)

    if use_filter:
        ind_subj = search_fn(rep_subj, rep_rel, rep_ent, pos_filter)
        ind_obj = search_fn(rep_ent, rep_rel, rep_obj, pos_filter)

    if corrupt == 's+o':
        stacked = torch.cat([
            torch.stack([rep_subj, rep_rel, rep_ent], dim=1),
            torch.stack([rep_ent, rep_rel, rep_obj], dim=1)
        ], dim=0)
    elif corrupt == 'o':
        stacked = torch.stack([rep_subj, rep_rel, rep_ent])
    else:
        stacked = torch.stack([rep_ent, rep_rel, rep_obj], dim=1)  # shape (n, 3, len(ent_for_corr))

    if not use_filter:
        return stacked.transpose(2, 1).reshape(-1, 3)
    else:
        return stacked.transpose(2, 1).reshape(-1, 3), ind_subj, ind_obj


def search_fn(subj, rel, obj, pos_filter):
    """
    Search for incoming or triplet into positives.
    :param subj: subjects in corrupted triplet.
    :param rel: objects in corrupted triplet.
    :param obj: relations in corrupted triplet.
    :param pos_filter: Dict of all positive triplets {(s_idx, p_idx, o_idx) : True.
    :return: torch.Tensor of FN indices.
    """
    indicies = [i for i, (s, p, o) in enumerate(zip(subj, rel, obj)) if (s.item(), p.item(), o.item()) in pos_filter]
    return torch.Tensor(indicies).to(subj.device).int()

