import numpy as np
import torch
import pickle
import os

from typing import List, Tuple


class DataMapper:
    def __init__(self,
                 triplets: List[Tuple[str]]
                 ):
        """
        Fits mapper to current data
        :param triplets: List of (subject, predicate, object) items.
        :returns None
        """
        if len(triplets) == 0:
            raise ValueError("Trying to initialize Mapper with empty data!")
        if len(triplets[0]) != 3:
            raise ValueError("Triplets must contain 3 items!")
        self.ent2idx = {}
        self.rel2idx = {}

        unique_ents = set()
        unique_rels = set()
        for s, p, o in triplets:
            unique_ents.update([s, o])
            unique_rels.add(p)

        for i, ent in enumerate(unique_ents):
            self.ent2idx[ent] = i

        for i, rel in enumerate(unique_rels):
            self.rel2idx[rel] = i

        self.idx2ent = {v: k for k, v in self.ent2idx.items()}
        self.idx2rel = {v: k for k, v in self.rel2idx.items()}

    def get_entity(self, idx):
        return self.idx2ent[idx]

    def get_relation(self, idx):
        return self.idx2rel[idx]

    def get_entity_idx(self, ent):
        return self.ent2idx[ent]

    def get_relation_idx(self, rel):
        return self.rel2idx[rel]

    def n_entities(self):
        return len(self.ent2idx)

    def n_relations(self):
        return len(self.rel2idx)

    def transform(self, triplets: List[str], return_tensors: str = 'pt'):
        """
        Transformes triplets to given/loaded indexes in init.
        :param triplets: List of triplet items of (subject, predicate, object).
        :param return_tensors: Type of tensor to return : 'np' for np.array, 'pt' for torch.Tensor,
            'list' for List[List[int]].
        :return: matrix of transformed triplets indexes.
        """
        if return_tensors not in ['pt', 'np', 'list']:
            raise TypeError("Unexpected type of tensor to return! Must be one of ['pt', 'np', 'list'] !")
        data = [(self.ent2idx[s], self.rel2idx[p], self.ent2idx[o]) for s, p, o in triplets]
        if return_tensors == 'pt':
            data = torch.Tensor(data)
        elif return_tensors == 'np':
            data = np.array(data)
        return data

    def save(self, path):
        """
        Dumps entity2idx and relation2idx mappings
        :param path: path to directory where to save mappings. If it's not exist, creates it.
        :return: None
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "entities.pkl"), "wb") as f:
            pickle.dump(self.ent2idx, f)

        with open(os.path.join(path, "relations.pkl"), "wb") as f:
            pickle.dump(self.rel2idx, f)

    def load(self, path):
        """
        Loads previous dumping
        :param path: path to directory with entities and relations.
        :return: loaded DataMapper class
        """
        ent_path = os.path.join(path, "entities.pkl")
        rel_path = os.path.join(path, "relations.pkl")
        if not os.path.exists(ent_path):
            raise RuntimeError(f"There is no entities.pkl file in directory {path}!")
        elif not os.path.exists(ent_path):
            raise RuntimeError(f"There is no relations.pkl file in directory {path}!")

        with open(ent_path, 'rb') as f:
            self.ent2idx = pickle.load(f)
            self.idx2ent = {k: v for v, k in self.ent2idx.items()}

        with open(rel_path, 'rb') as f:
            self.rel2idx = pickle.load(f)
            self.idx2rel = {k: v for v, k in self.rel2idx.items()}

        return self
