import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.gamma_pos > 0:
                pt0 = xs_pos * y
                loss_pos = -((1 - pt0) ** self.gamma_pos) * los_pos
            else:
                loss_pos = -los_pos
                
            if self.gamma_neg > 0:
                pt1 = xs_neg * (1 - y)  # pt = p if t = 1 else 1-p
                loss_neg = -((pt1) ** self.gamma_neg) * los_neg
            else:
                loss_neg = -los_neg
                
            loss = loss_pos + loss_neg

        loss = -loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False, reduction='mean'):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            if self.gamma_pos > 0:
                xs_pos = xs_pos * y
                xs_pos.add_(self.eps).pow_(self.gamma_pos).mul_(los_pos)
            else:
                xs_pos = los_pos
            if self.gamma_neg > 0:
                xs_neg = xs_neg * (1 - y)
                xs_neg.add_(self.eps).pow_(self.gamma_neg).mul_(los_neg)
            else:
                xs_neg = los_neg
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            asymmetric_w = torch.add(xs_pos, xs_neg)
        else:
            asymmetric_w = torch.add(los_pos, los_neg)

        asymmetric_w = -asymmetric_w
        
        if self.reduction == 'mean':
            return asymmetric_w.mean()
        elif self.reduction == 'sum':
            return asymmetric_w.sum()
        else:  # 'none'
            return asymmetric_w