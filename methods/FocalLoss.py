import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

class BaselineLoss(_Loss):
    """Weighted focal loss
    See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
    a good explanation
    Attributes
    ----------
    alpha: torch.tensor of size 8, class weights
    gamma: torch.tensor of size 1, positive float, for gamma = 0 focal loss is
    the same as CE loss, increases gamma reduces the loss for the "hard to classify
    examples"
    """

    def __init__(
        self,
        alpha=torch.tensor(
            [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
        ),
        gamma=2.0,
    ):
        super(BaselineLoss, self).__init__()
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Weighted focal loss function
        Parameters
        ----------
        inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
        targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
        """
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()

