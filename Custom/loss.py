import torch

from recbole.model.loss import BPRLoss

class MaskedBPRLoss(BPRLoss):
    """ MaskedBPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N, size)
        - Neg_score: (N, size), same shape as the Pos_score
        - Mask: (N, size), boolean
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(MaskedBPRLoss, self).__init__(gamma)
        self.gamma = gamma

    def forward(self, pos_score, neg_score, mask=1):
        loss = -torch.div(torch.mul(mask, torch.log(self.gamma + torch.sigmoid(pos_score - neg_score))), mask.sum(dim=1, keepdims=True))
        return loss.mean(dim=0).sum()
