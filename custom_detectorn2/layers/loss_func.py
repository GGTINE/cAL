import torch
import torch.nn.functional as F
from detectron2.utils.comm import get_world_size
from custom_detectorn2.utils.comm import reduce_sum
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from custom_detectorn2.utils.integral import Integral
from custom_detectorn2.layers import IOULoss, NLLoss


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
        top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
    )
    return torch.sqrt(ctrness)


class ClassificationLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits_pred, class_target):
        class_loss_all = sigmoid_focal_loss_jit(
            logits_pred,
            class_target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="none",
        )
        class_loss = class_loss_all.sum()

        return class_loss


class RegressionLoss(nn.Module):
    def __init__(self, loc_loss_type, kl_loss_weight, reg_max, reg_discrete, tsbetter_reg, tsbetter_reg_cert):
        super().__init__()
        self.reg_loss_func = NLLoss()
        self.loc_loss_func = IOULoss(loc_loss_type)
        self.kl_loss_weight = kl_loss_weight
        self.discrete = reg_discrete
        self.ts_reg = tsbetter_reg
        self.ts_reg_cert = tsbetter_reg_cert

        self.integral = Integral(reg_max)

    def forward(self, reg_pred, reg_pred_std, reg_targets, boundary_vars=None):
        # reg_discrete
        if self.discrete:
            reg_pred = self.integral(reg_pred)

        if boundary_vars is not None:
            loc_conf_student = 1 - reg_pred_std.sigmoid()
            loc_conf_teacher = 1 - boundary_vars.sigmoid()
            select = (loc_conf_teacher > self.ts_reg_cert) * (loc_conf_teacher > loc_conf_student + self.ts_reg)

            reg_student = reg_pred
            reg_teacher = reg_targets

            if select.sum() > 0:
                reg_loss = F.smooth_l1_loss(reg_student[select], reg_teacher[select], beta=0.0)
            else:
                reg_loss = torch.tensor(0).to(device=reg_pred.device, dtype=torch.float32)
        else:
            ctrness_targets = compute_ctrness_targets(reg_targets)

            loss_denorm = max(
                reduce_sum(ctrness_targets.sum()).item() / get_world_size(), 1e-6
            )

            iou, iou_loss = self.loc_loss_func(
                reg_pred, reg_targets, ctrness_targets, loss_denorm
            )
            nl_loss = self.reg_loss_func(
                reg_pred,
                reg_pred_std,
                reg_targets,
                iou_weight=iou.detach(),
            )

            reg_loss = self.kl_loss_weight * nl_loss + iou_loss

        return reg_loss


class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reg_targets, ctrness_pred):
        ctrness_targets = compute_ctrness_targets(reg_targets)

        ctrness_loss = F.binary_cross_entropy_with_logits(
            ctrness_pred, ctrness_targets, reduction="sum"
        )

        return ctrness_loss
