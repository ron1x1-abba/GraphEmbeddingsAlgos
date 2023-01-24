import torch

class PairwiseLoss(torch.nn.Module):
    def __init__(self, margin: float):
        """
        Initialize Loss class.
        :param margin: Margin to use for Pairwise loss.
        """
        super(PairwiseLoss, self).__init__()

        self.margin = margin

    def forward(self,
                pos_scores: torch.Tensor,
                neg_scores: torch.Tensor
                ):
        return torch.maximum(self.margin - pos_scores + neg_scores, torch.Tensor([0])).sum()
