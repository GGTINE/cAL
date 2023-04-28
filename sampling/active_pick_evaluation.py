import os
import json
import numpy as np
import argparse
from collections import defaultdict, Counter


def norm_dict(_dict):
    v_max = max(_dict.values())
    for k, v in _dict.items():
        _dict[k] = v / v_max
    _dict["max"] = v_max
    return _dict


def preprocess_data(file_name):
    print("Loading File {}...".format(file_name))
    with open(file_name, "r") as f:
        data = json.load(f)
    difficult_indicators = defaultdict(float)  # entropy
    center_indicators = defaultdict(float)  # centerness
    cls_conf = defaultdict(float)  # cls conf
    score = defaultdict(float)  # cls_n_ctr conf
    diversity_indicators = defaultdict(list)  # pred_class
    pred_bbox_indicators = defaultdict(list)

    for index, (image_path, boxes_info) in enumerate(data.items()):
        _difficult = 0
        _center = 0
        _cls_conf = 0
        _cls_n_ctr_conf = 0
        _diversity = 0

        _cls_set = set()
        _bbox = []

        for box in boxes_info:
            _difficult = box["entropy"]
            _center += box["center"]
            _cls_conf += box["confidence score"]
            _cls_n_ctr_conf += box["score"]
            _cls_set.add(box["pred class"])
            _bbox.append(box["pred box"])
        # _diversity = len(_cls_set)

        difficult_indicators[image_path] = _difficult
        center_indicators[image_path] = _center
        cls_conf[image_path] = _cls_conf
        score[image_path] = _cls_n_ctr_conf
        diversity_indicators[image_path] = list(_cls_set)
        pred_bbox_indicators[image_path] = _bbox

    return data, difficult_indicators, center_indicators, cls_conf, score, diversity_indicators, pred_bbox_indicators


def combine_metrics(
    data,
    difficult_indicators,
    center_indicators,
    confidence,
    diversity_indicators,
    weights,
    diverse_weight,
    file_name,
):
    f = open(file_name + ".txt", "w")
    final_value = defaultdict(float)
    for image_path, _ in data.items():
        _final_value = (
            difficult_indicators[image_path] * weights["entropy"]
            + center_indicators[image_path] * weights["center"]
            + confidence[image_path] * weights["confidence"]
            + diversity_indicators[image_path] * weights["diversity"]
        )
        final_value[image_path] = _final_value
        f.write(str(_final_value) + "\n")
    f.close()

    with open(file_name + ".json", "w") as f:
        f.write(json.dumps(final_value))

    print("Finish {}".format(file_name))


def special_(args):
    weights = {
        "entropy": 1.0,  # 높을 수록
        "center": 1.0,  # 낮을 수록
        "confidence": 1.0,  # cls_conf와 score 평균이 낮을 수록
        "diversity": 1.0,  # weight와 곱한 것이 높을 수록
        "iou": 1.0,  # 낮을 수록.. 이건 변경이 필요해보임
    }  # weights to combine metrics

    (
        data,
        difficult_indicators,
        center_indicators,
        cls_conf,
        score,
        diversity_indicators,
        pred_bbox_indicators,
    ) = preprocess_data(file_name=args.original_file)

    _difficult_indicators = norm_dict(difficult_indicators)
    _center_indicators = norm_dict(center_indicators)

    confidence = {}
    for key in [k for k in cls_conf.keys() if k in score]:
        confidence[key] = (cls_conf[key] + score[key]) / 2
    confidence = norm_dict(confidence)

    (
        aug_data,
        aug_difficult_indicators,
        aug_center_indicators,
        aug_cls_conf,
        aug_score,
        aug_diversity_indicators,
        aug_pred_bbox_indicators,
    ) = preprocess_data(file_name=args.augmentation_file)

    # label data 불러와서 class 분포 확인 후 클래스 별 weight 부여 후
    # diversity의 클래스 * weight 해서 점수화 하기

    # Consistency 구해서 box iou를 점수로 사용
    # _diversity_indicators = norm_dict(diversity_indicators)

    combine_metrics(
        data,
        _difficult_indicators,
        _center_indicators,
        confidence,
        # _diversity_indicators,
        weights=weights,
        # diverse_weight=label_dist,
        file_name=args.indicator_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score function")
    parser.add_argument(
        "--original-file",
        type=str,
        default="./output/inference/original_score_factor.json",
    )
    parser.add_argument(
        "--augmentation-file",
        type=str,
        default="./output/inference/augmentation_score_factor.json",
    )
    parser.add_argument(
        "--indicator-file", type=str, default="./output/coco/2random_maxnorm"
    )  # indicator file to be used in picking data
    args = parser.parse_args()
    special_(args)
