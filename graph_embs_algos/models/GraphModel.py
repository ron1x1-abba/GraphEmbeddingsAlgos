import torch


class GraphModel(torch.nn.Module):
    def __init__(self,
                 n_relations: int,
                 n_entities: int,
                 emb_dim: int,
                 nonlinear: str = None
                 ):
        super(GraphModel, self).__init__()
        self.entity_embedding = torch.nn.Embedding(n_entities, emb_dim)
        self.relation_embedding = torch.nn.Embedding(n_relations, emb_dim)
        self.nonlinear = torch.nn.Tanh() if nonlinear == 'tanh' else torch.nn.Sigmoid()

    def forward(self):
        raise NotImplementedError("This function should be override in any Graph Algorithm")
        e_s = self.entity_embedding()
        e_p = self.relation_embedding()
        e_o = self.entity_embedding()
        return e_s, e_p, e_o

    def predict(self):
        """
        Predict the scores of triples using a trained embedding model.
        The function returns raw scores generated by the model.

        :param triplets: string triplets of (ent, rel, ent)
        :return: torch.Tensor of raw triplet scores
        """
        raise NotImplementedError("This function should be override in any Graph Algorithm")


    def predict_proba(self):
        raise NotImplementedError("This function should be override in any Graph Algorithm")
