import numpy as np

import torch
from torch import nn

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from custom_detectorn2.utils.detector_post import detector_postprocess


@META_ARCH_REGISTRY.register()
class OneStageDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        self.cfg = cfg

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(
        self,
        batched_inputs,
        output_raw=False,
        nms_method="cls_n_ctr",
        branch="label",
    ):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        # pseudo-labels for classification and regression
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif (
            "instances_class" in batched_inputs[0]
            and "instances_reg" in batched_inputs[0]
        ):
            gt_cls = [x["instances_class"].to(self.device) for x in batched_inputs]
            gt_reg = [x["instances_reg"].to(self.device) for x in batched_inputs]
            gt_instances = {"cls": gt_cls, "reg": gt_reg}
        else:
            gt_instances = None

        if branch == "unlabel":
            proposal_dict = {}
            for i in gt_instances:
                if i == "reg":
                    branch = i
                proposal, raw_pred = self.proposal_generator(
                    images,
                    features,
                    gt_instances[i],
                    output_raw=output_raw,
                    nms_method=nms_method,
                    branch=branch,
                )
                proposal_dict[i] = proposal
            return proposal_dict
        else:
            proposal, raw_pred = self.proposal_generator(
                images,
                features,
                gt_instances,
                output_raw=output_raw,
                nms_method=nms_method,
                branch=branch,
            )

        # training
        if self.training:
            return proposal

        # inference
        if output_raw:
            return proposal, raw_pred
        else:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                proposal, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"proposals": r})

            processed_results = [
                {"instances": r["proposals"]} for r in processed_results
            ]
            return processed_results
