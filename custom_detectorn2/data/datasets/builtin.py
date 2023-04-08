# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {
    "coco": {
        "coco_2017_unlabel": (
            "memcache_manifold://mobile_vision_dataset/tree/coco_unlabel2017",
            "memcache_manifold://mobile_vision_dataset/tree/coco_unlabel2017/coco_jsons/image_info_unlabeled2017.json",
        ),
        "coco_2017_for_voc20": (
            "coco",
            "coco/annotations/google/instances_unlabeledtrainval20class.json",
        ),
    }
}


def register_coco_unlabel():
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(key, meta, json_file, image_root)


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    dataset_dicts = []

    for img_dict in imgs:
        record = {
            "file_name": os.path.join(image_root, img_dict["file_name"]),
            "height": img_dict["height"],
            "width": img_dict["width"],
        }
        dataset_dicts.append(record)

    return dataset_dicts


register_coco_unlabel()
