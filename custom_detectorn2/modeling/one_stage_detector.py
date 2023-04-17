import numpy as np
import torch
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from torch import nn


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    In addition to the post processing of detectron2, we add scalign for
    bezier control points.
    """
    scale_x, scale_y = (
        output_width / results.image_size[1],
        output_height / results.image_size[0],
    )
    results = d2_postprocesss(results, output_height, output_width, mask_threshold)

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)

    return results


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

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(
        self,
        batched_inputs,
        output_raw=False,
        nms_method="cls_n_ctr",
        branch="labeled",
    ):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        # pseudo-labels for classification and regression
        if "instances" in batched_inputs[0] and branch != "teacher_weak":
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "instances_class" in batched_inputs[0] and "instances_reg" in batched_inputs[0]:
            gt_cls = [x["instances_class"].to(self.device) for x in batched_inputs]
            gt_reg = [x["instances_reg"].to(self.device) for x in batched_inputs]
            gt_instances = {"cls": gt_cls, "reg": gt_reg}
        else:
            gt_instances = None

        # training
        if self.training:
            proposal_losses = self.proposal_generator(
                images,
                features,
                gt_instances,
                nms_method=nms_method,
                branch=branch,
            )
            return proposal_losses
        # inference
        else:
            if output_raw:
                proposals, raw_pred = self.proposal_generator(
                    images,
                    features,
                    gt_instances,
                    output_raw=True,
                    nms_method=nms_method,
                    branch=branch,
                )
                return proposals, raw_pred
            else:
                proposals = self.proposal_generator(
                    images,
                    features,
                    gt_instances,
                    output_raw=False,
                    nms_method=nms_method,
                    branch=branch,
                )
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    proposals, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"proposals": r})

                processed_results = [
                    {"instances": r["proposals"]} for r in processed_results
                ]
                return processed_results
