import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Class Classification.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

    Args:
        alpha (float/list): Weighting factor for each class.
                            If float, applied to rare class (usually not used in multi-class this way).
                            If list, weights per class.
        gamma (float): Focusing parameter. Higher = focus more on hard examples.
        reduction (str): 'mean' or 'sum'.
        ignore_index (int): Class to ignore.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1-alpha]) # binary case assumption
            else:
                self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        inputs: (N, C) logits
        targets: (N) class indices
        """
        if inputs.ndim > 2:
            # (N, C, d1, d2) -> (N*d1*d2, C)
            c = inputs.shape[1]
            inputs = inputs.permute(0, *range(2, inputs.ndim), 1).reshape(-1, c)
            targets = targets.view(-1)

        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)

        log_pt = log_pt.gather(1, targets.unsqueeze(1))
        log_pt = log_pt.view(-1)
        pt = pt.gather(1, targets.unsqueeze(1)).view(-1)

        # Focal term
        loss = -1 * (1 - pt) ** self.gamma * log_pt

        # Alpha weighting
        if self.alpha is not None:
             if self.alpha.device != inputs.device:
                 self.alpha = self.alpha.to(inputs.device)
             at = self.alpha.gather(0, targets.view(-1))
             loss = loss * at

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class UniversalLoss(nn.Module):
    """
    Combined Loss for Universal Model:
    1. Direction (Focal Loss)
    2. Volatility (MSE/Huber)
    """
    def __init__(self, num_classes=3, gamma=2.0, volatility_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.mse = nn.MSELoss()
        self.vol_weight = volatility_weight

    def forward(self, predictions, targets):
        """
        predictions: (pred_dir_logits, pred_vol)
        targets: (target_dir_class, target_vol)
        """
        pred_dir, pred_vol = predictions
        target_dir = targets[:, 0].long() # Class index (0, 1, 2)
        target_vol = targets[:, 1].unsqueeze(1) # Value

        loss_dir = self.focal(pred_dir, target_dir)
        loss_vol = self.mse(pred_vol, target_vol)

        return loss_dir + (self.vol_weight * loss_vol)
