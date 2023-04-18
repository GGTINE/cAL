# most part of this file is modified from AdelaiDet
# https://github.com/aim-uofa/AdelaiDet


import logging

import torch
from detectron2.layers import cat
from detectron2.structures import Boxes, Instances
from torch import nn
from custom_detectorn2.layers import ml_nms
from custom_detectorn2.utils.integral import Integral

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: minibatch 내 이미지 수
    L: RPN에서 이미지 당 feature map 수
    Hi, Wi: i-번째 이미지의 height, width
    4: 박스 크기

Naming convention:
    labels: refers to the ground-truth class of an position.
    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.
    logits_pred: predicted classification scores in [-inf, +inf];
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets
    ctrness_pred: predicted centerness scores
"""


class FCOSOutputs(nn.Module):
    def __init__(self, cfg):
        super(FCOSOutputs, self).__init__()

        self.cfg = cfg

        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.strides = cfg.MODEL.FCOS.FPN_STRIDES

        # bin offset classification
        self.reg_discrete = cfg.MODEL.FCOS.REG_DISCRETE

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.reg_max = cfg.MODEL.FCOS.REG_MAX

        self.integral = Integral(self.reg_max)

    def prepare_instance(
        self,
        logits_pred,
        reg_pred,
        ctrness_pred,
        locations,
        gt_instances,
        reg_pred_std,
        branch,
    ):
        training_targets = self._get_ground_truth(locations, gt_instances, branch)

        instances = Instances((0, 0))
        instances.labels = cat(
            [x.reshape(-1) for x in training_targets["labels"]], dim=0
        )
        instances.reg_targets = cat(
            [x.reshape(-1, 4) for x in training_targets["reg_targets"]], dim=0
        )
        instances.locations = cat(
            [x.reshape(-1, 2) for x in training_targets["locations"]], dim=0
        )
        instances.logits_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits_pred],
            dim=0,
        )
        instances.ctrness_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1) for x in ctrness_pred], dim=0
        )

        if self.reg_discrete:
            instances.reg_pred = cat(
                [
                    x.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
                    for x in reg_pred
                ],
                dim=0,
            )
        else:
            instances.reg_pred = cat(
                [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred], dim=0
            )

        if self.cfg.MODEL.FCOS.KL_LOSS:
            assert reg_pred_std is not None
            instances.reg_pred_std = cat(
                [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred_std], dim=0
            )

        if self.cfg.SEMISUPNET.CONSIST_REG_LOSS == "ts_locvar_better_nms_nll_l1" and branch == "reg":
            instances.boundary_vars = cat(
                [x.reshape(-1, 4) for x in training_targets["boundary_vars"]],
                dim=0,
            )

        return instances

    # other functions
    def _transpose(self, training_targets, num_loc_list):
        """
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        """
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(torch.cat(targets_per_level, dim=0))
        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances, branch):
        num_loc_list = [len(loc) for loc in locations]
        # compute locations to size ranges
        loc_to_size_range = []
        for lo, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(
                self.sizes_of_interest[lo]
            )
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[lo], -1)
            )
            # [prev_layer_size, this layer_size ]
            # [[-1,64], .... ,[64,128],...,[128,256], ...,[256,512],... [512,100000]]

        loc_to_size_range = torch.cat(
            loc_to_size_range, dim=0
        )  # size [L1+L2+...+L5, 2]
        locations = torch.cat(locations, dim=0)  # size [L1+L2+...+L5, 2]

        # compute the reg, label target for each element
        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, branch,
        )

        training_targets["locations"] = [
            locations.clone() for _ in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        # we normalize reg_targets by FPN's strides here
        # reg_targets is normalized for each level!
        #  this is ltrb format
        reg_targets = training_targets["reg_targets"]
        for la in range(len(reg_targets)):
            reg_targets[la] = reg_targets[la] / float(self.strides[la])

        return training_targets

    def compute_targets_for_locations(self, locations, targets, size_ranges, branch):
        labels = []
        reg_targets = []
        boundary_vars = []

        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):  # image-wise operation
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # box weight weights
            if branch == "reg":
                boundary_var_per_im = targets_per_im.reg_pred_std

            # no gt
            if bboxes.numel() == 0:
                # no bboxes then all labels are background
                labels.append(
                    labels_per_im.new_zeros(locations.size(0)) + self.num_classes
                )
                # no bboxes then all boxes weights are zeros
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                if branch == "reg":
                    boundary_vars.append(locations.new_zeros((locations.size(0), 4)))
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            # filter out these box is too small or too big for each scale
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = (max_reg_targets_per_im >= size_ranges[:, [0]]) & (
                max_reg_targets_per_im <= size_ranges[:, [1]]
            )

            # compute the area for each gt box
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            # set points (outside box/small region) as background
            locations_to_gt_area[is_in_boxes == 0] = INF
            # set points with too large displacement or too small displacement as background
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(
                dim=1
            )

            # use the minial area as creteria to choose ground-truth boxes of regression for each point
            reg_targets_per_im = reg_targets_per_im[
                range(len(locations)), locations_to_gt_inds
            ]

            # regard object in different image as different instance

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            if branch == "reg":
                boundary_var_per_im = boundary_var_per_im[locations_to_gt_inds]
                boundary_var_per_im[locations_to_min_area == INF] = 99999.0
                boundary_vars.append(boundary_var_per_im)

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return_dict = {
            "labels": labels,
            "reg_targets": reg_targets,
        }
        if branch == "reg":
            return_dict["boundary_vars"] = boundary_vars

        return return_dict

    def predict_proposals(
        self,
        logits_pred,
        reg_pred,
        ctrness_pred,
        locations,
        image_sizes,
        reg_pred_std=None,
        nms_method="cls_n_ctr",
    ):

        if self.training:
            self.pre_nms_thresh = self.cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
            self.pre_nms_topk = self.cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
            self.post_nms_topk = self.cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        else:
            self.pre_nms_thresh = self.cfg.MODEL.FCOS.INFERENCE_TH_TEST
            self.pre_nms_topk = self.cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
            self.post_nms_topk = self.cfg.MODEL.FCOS.POST_NMS_TOPK_TEST

        sampled_boxes = []

        bundle = {
            "l": locations,
            "o": logits_pred,
            "r": reg_pred,
            "c": ctrness_pred,
            "s": self.strides,
        }

        if reg_pred_std is not None:
            bundle["r_std"] = reg_pred_std

        # each iteration = 1 scale
        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.

            l = per_bundle["l"]
            o = per_bundle["o"]

            if self.reg_discrete:  # discrete to scalar
                bs = per_bundle["r"].shape[0]
                imgw = per_bundle["r"].shape[2]
                imgh = per_bundle["r"].shape[3]
                reg_discre_raw = (
                    per_bundle["r"]
                    .permute(0, 2, 3, 1)
                    .reshape(-1, 4 * (self.reg_max + 1))
                )
                scalar_r = self.integral(reg_discre_raw).reshape(bs, imgw, imgh, 4)
                scalar_r = scalar_r.permute(0, 3, 1, 2)
                r = scalar_r * per_bundle["s"]

                r_cls = (per_bundle["r"], per_bundle["s"])
            else:
                r = per_bundle["r"] * per_bundle["s"]
                r_cls = None

            c = per_bundle["c"]

            r_std = per_bundle["r_std"] if "r_std" in bundle else None

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, r_cls, c, image_sizes, r_std, nms_method
                )
            )

        # nms
        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def forward_for_single_feature_map(
        self,
        locations,
        logits_pred,
        reg_pred,
        reg_pred_cls,
        ctrness_pred,
        image_sizes,
        reg_pred_std=None,
        nms_method="cls_n_ctr",
    ):
        N, C, H, W = logits_pred.shape
        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()

        if reg_pred_cls is not None:
            box_reg_cls = (
                reg_pred_cls[0]
                .view(N, 4 * (self.reg_max + 1), H, W)
                .permute(0, 2, 3, 1)
            )
            box_reg_cls = box_reg_cls.reshape(N, -1, 4 * (self.reg_max + 1))
            scalar = reg_pred_cls[1]

        if reg_pred_std is not None:
            box_regression_std = reg_pred_std.view(N, 4, H, W).permute(0, 2, 3, 1)
            box_regression_std = box_regression_std.reshape(N, -1, 4)

        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)
        cls_confs = logits_pred

        if nms_method == "cls":
            logits_pred = logits_pred
        elif nms_method == "cls_n_loc":
            assert box_regression_std is not None
            boundary_regression_std = 1 - box_regression_std.sigmoid()
            box_reg_std = boundary_regression_std.mean(2)
            logits_pred = logits_pred * box_reg_std[:, :, None]
        else:  # default cls + ctr
            logits_pred = logits_pred * ctrness_pred[:, :, None]

        results = []
        for i in range(N):  # each image
            # select pixels larger than threshold (0.05)
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            # get the index of pixel and its class prediction
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            # for bin classification
            if reg_pred_cls is not None:
                per_box_reg_cls = box_reg_cls[i]
                per_box_reg_cls = per_box_reg_cls[per_box_loc]

            # for localization std
            if reg_pred_std is not None:
                per_box_regression_std = box_regression_std[i]
                per_box_regression_std = per_box_regression_std[per_box_loc]

            # centerness
            per_centerness = ctrness_pred[i]
            per_centerness = per_centerness[per_box_loc]
            per_cls_conf = cls_confs[i]
            per_cls_conf = per_cls_conf[per_candidate_inds]

            # select top k
            per_pre_nms_top_n = pre_nms_top_n[i]

            # check whether per_candidate boxes is too many
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(
                    per_pre_nms_top_n, sorted=False
                )
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

                if reg_pred_cls is not None:
                    per_box_reg_cls = per_box_reg_cls[top_k_indices]

                if reg_pred_std is not None:
                    per_box_regression_std = per_box_regression_std[top_k_indices]

                per_centerness = per_centerness[top_k_indices]
                per_cls_conf = per_cls_conf[top_k_indices]

            detections = torch.stack(
                [
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 0] + per_box_regression[:, 2],
                    per_locations[:, 1] + per_box_regression[:, 3],
                ],
                dim=1,
            )

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            if nms_method == "cls_n_ctr" or nms_method == "cls_n_loc":
                boxlist.scores = torch.sqrt(per_box_cls)
            elif nms_method == "cls" or nms_method == "ctr":
                boxlist.scores = per_box_cls
            else:
                raise ValueError("Undefined nms criteria")

            if reg_pred_cls is not None:
                boxlist.reg_pred_cls = per_box_reg_cls
                boxlist.reg_pred_cls_scalar = (
                    torch.ones(per_box_reg_cls.shape[0]).to(device=logits_pred.device)
                    * scalar
                )

            if reg_pred_std is not None:
                boxlist.reg_pred_std = per_box_regression_std

            # boxlist.scores = torch.sqrt(per_box_cls)
            # boxlist.scores = torch.sqrt(per_box_cls)

            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            boxlist.centerness = per_centerness
            boxlist.cls_confid = per_cls_conf

            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.cfg.MODEL.FCOS.NMS_TH)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores, number_of_detections - self.post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
