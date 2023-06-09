# This file is from AdelaiDet
# https://github.com/aim-uofa/AdelaiDet

import torch
from torch import nn


class KLLoss(nn.Module):
    """Kullback-Leibler Divergence Loss for Regression"""

    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, input, input_std, target, weight=None, beta=1.0, loss_denorm=None, method="weight_ctr_sum"):
        if beta < 1e-5:
            # if beta == 0, then torch.where will result in nan gradients when
            # the chain rule is applied due to pytorch implementation details
            # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
            # zeros, rather than "no gradient"). To avoid this issue, we define
            # small values of beta to be exactly l1 loss.
            loss = torch.abs(input - target)
        else:
            n = torch.abs(input - target)
            cond = n < beta
            l1_smooth = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)
            # loss = torch.exp(-input_std)*l1_smooth.detach() + 0.5*input_std + l1_smooth
            loss = torch.exp(-input_std) * l1_smooth + 0.5 * input_std

        if method == "weight_ctr_sum":
            assert weight is not None
            loss = loss.sum(dim=1)
            return (loss * weight).sum()
        elif method == "weight_ctr_mean":
            assert weight is not None
            assert loss_denorm is not None
            loss = loss.sum(dim=1)
            return (loss * weight).sum() / loss_denorm
        elif method == "sum":
            return loss.sum()
        elif method == "mean":
            return loss.mean()
        else:
            raise ValueError("No defined regression loss method")


class NLLoss(nn.Module):
    def __init__(self):
        super(NLLoss, self).__init__()

    def forward(self, input, input_std, target, iou_weight=None):
        sigma_sq = torch.square(input_std.sigmoid())
        first_term = torch.square(target - input) / (2 * sigma_sq)
        second_term = 0.5 * torch.log(sigma_sq)
        sum_before_iou = (first_term + second_term).sum(dim=1) + 2 * torch.log(2 * torch.Tensor([torch.pi]).cuda())
        loss_mean = (sum_before_iou * iou_weight).mean()

        return loss_mean
