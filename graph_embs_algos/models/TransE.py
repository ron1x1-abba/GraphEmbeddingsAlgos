
import torch
from .GraphModel import GraphModel
from typing import Union


class TransE(GraphModel):
    def __init__(self,
                 n_relations: int,
                 n_entities: int,
                 emb_dim: int,
                 nonlinear: str = None,
                 norm_type: Union[int, str] = 2
                 ):
        """
        :param n_relations: Number of relations on which model was trained.
        :param n_entities: Number of entities on which model was trained.
        :param emb_dim: Embeddings size.
        :param nonlinear: Type of nonlinearity. Can be one of ['tanh', 'sigmoid'], if None returns linear.
        :param norm_type: Order of normalization embeddings. Can be one of ["cos", "fro", "nuc", 1, 2, inf, -inf].
        """
        super(TransE, self).__init__(
            n_relations=n_relations,
            n_entities=n_entities,
            emb_dim=emb_dim
        )
        self.norm_type = norm_type
        if nonlinear is None:
            self.nonlinear = None
        elif nonlinear == 'tanh':
            self.nonlinear = torch.nn.TanH()
        elif nonlinear == 'sigmoid':
            self.nonlinear = torch.nn.Sigmoid()
        else:
            raise ValueError(f"Nonlinearity {nonlinear} is not implemented!")

    def forward(self, subjects, predicates, objects):
        e_o = self.entity_embedding(objects)
        e_p = self.relation_embedding(predicates)
        e_s = self.entity_embedding(subjects)
        dist = e_s + e_p - e_o
        normalized = torch.linalg.norm(dist, ord=self.norm_type, dim=1)
        if self.nonlinear is not None:
            return
        return -normalized

    def predict(self, objects, predicates, subjects):
        norm = self.forward(objects, predicates, subjects)
        if self.nonlinear is not None:
            return self.nonlinear(norm)
        return norm
