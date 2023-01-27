
import torch
from .GraphModel import GraphModel
from typing import Union

class RotatE(GraphModel):
    def __init__(self,
                 n_relations: int,
                 n_entities: int,
                 emb_dim: int,
                 nonlinear: str = None,
                 norm_type: Union[int, str] = 2,
                 gamma: float = 12.0
                 ):
        """
        Implementation of RotatE model.

        :param n_relations: Number of relations on which model was trained.
        :param n_entities: Number of entities on which model was trained.
        :param emb_dim: Embeddings size.
        :param nonlinear: Type of nonlinearity. Can be one of ['tanh', 'sigmoid'], if None returns linear.
        :param norm_type: Order of normalization embeddings. Can be one of ["cos", "fro", "nuc", 1, 2, inf, -inf].
        :param gamma:
        """
        super(RotatE, self).__init__(
            n_relations=n_relations,
            n_entities=n_entities,
            emb_dim=emb_dim,
            double_ent_dim=True,
            nonlinear=nonlinear,
        )
        self.norm_type = norm_type
        self.emb_dim = emb_dim
        if nonlinear is None:
            self.nonlinear = None
        elif nonlinear == 'tanh':
            self.nonlinear = torch.nn.TanH()
        elif nonlinear == 'sigmoid':
            self.nonlinear = torch.nn.Sigmoid()
        else:
            raise ValueError(f"Nonlinearity {nonlinear} is not implemented!")
        self.epsilon = 2.0
        self.gamma = gamma
        embedding_range = (gamma + self.epsilon) / emb_dim
        torch.nn.init.uniform_(self.entity_embedding.weight, embedding_range, embedding_range)
        torch.nn.init.uniform_(self.relation_embedding.weight, embedding_range, embedding_range)

    def forward(self, subjects, predicates, objects):
        e_o = self.entity_embedding(objects)
        re_e_o, im_e_o = e_o.view(-1, self.emb_dim, 2).permute(2, 0, 1)
        e_p = self.relation_embedding(predicates)
        re_e_p, im_e_p = torch.cos(e_p), torch.sin(e_p)
        e_s = self.entity_embedding(subjects)
        re_e_s, im_e_s = e_s.view(-1, self.emb_dim, 2).permute(2, 0, 1)
        dist = torch.stack([
            re_e_s * re_e_p - im_e_s * im_e_p - re_e_o,
            re_e_s * im_e_p + im_e_s * re_e_p - im_e_o
        ], dim=0)
        normalized = dist.norm(p=self.norm_type, dim=0).sum(1)
        if self.nonlinear is not None:
            return self.gamma - self.nonlinear(normalized)
        return self.gamma - normalized

    def predict(self, objects, predicates, subjects):
        norm = self.forward(objects, predicates, subjects)
        return norm
