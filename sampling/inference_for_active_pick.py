import json
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data import detection_utils as utils
from detectron2.structures import Boxes
from detectron2.utils.file_io import PathManager

from custom_detectorn2.modeling.pseudo_generator import PseudoGenerator
from custom_detectorn2.utils.visual import visual_img
from custom_detectorn2.config import add_ubteacher_config
from custom_detectorn2.engine.trainer import ActiveTrainer
from custom_detectorn2.modeling.meta_arch.ts_ensemble import EnsembleTSModel

import warnings

warnings.filterwarnings("ignore")


@torch.no_grad()
def uncertainty_entropy(predict):
    # p.size() = num_instances of an image, num_classes
    uncertainty_list = []
    for p in predict:
        # p = F.softmax(p, dim=1)
        p = p.sigmoid()
        p = -torch.log2(p) * p
        entropy_instances = torch.sum(p)

        # set uncertainty of image eqs the mean uncertainty of instances
        uncertainty_list.append(entropy_instances.to(device="cpu"))
    return np.mean(uncertainty_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    assert args.eval_only is True, "Inference should be eval only."
    inference(ActiveTrainer, cfg, option="label")


# cls_hook, ctr_hook = [], []


# def cls_predictor_hooker(module, input, output):
#     cls_hook.append(output.clone().detach().cpu())
#
#
# def ctr_predictor_hooker(module, input, output):
#     ctr_hook.append(output.clone().detach().cpu())


@torch.no_grad()
def inference(trainer, cfg, option):
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DPATASETS.PROOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if option == "label":
        print("prepare label weight")
        label_weight = prepare_label_weight(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )
        return
    print("Loading Model named: ", cfg.MODEL.WEIGHTS)
    model = trainer.build_model(cfg)
    model_teacher = trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)
    pseudo_generator = PseudoGenerator(cfg)

    DetectionCheckpointer(ensem_ts_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    ensem_ts_model.modelTeacher.eval()
    ensem_ts_model.modelTeacher.training = False

    # cls_logits, bbox_pred, ctrness
    # ensem_ts_model.modelTeacher.proposal_generator.fcos_head.cls_logits.register_forward_hook(cls_predictor_hooker)
    # ensem_ts_model.modelTeacher.proposal_generator.fcos_head.ctrness.register_forward_hook(ctr_predictor_hooker)
    dic = {}
    aug_dic = {}

    for j, item in enumerate(dataset_dicts):
        file_name = item["file_name"]
        print(j, file_name)

        image = utils.read_image(file_name, format="BGR")
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)
        aug_image = consistency_transform(image)

        pseudo_label, num_instance = create_pseudo(
            ensem_ts_model, image, pseudo_generator
        )
        aug_pseudo_label, aug_num_instance = create_pseudo(
            ensem_ts_model,
            aug_image,
            pseudo_generator,
        )
        aug_image, aug_pseudo_label[0].gt_boxes, aug_pseudo_label[0].locations = horizontal_flip(
            aug_image, aug_pseudo_label[0].gt_boxes.tensor, aug_pseudo_label[0].locations, aug_num_instance
        )

        file_name = file_name.split("/")[-1].split(".")[0]
        dic = record_score_dict(pseudo_label, num_instance, dic, file_name)
        aug_dic = record_score_dict(aug_pseudo_label, aug_num_instance, aug_dic, file_name)

        # Multi-prediction
        # entropy = uncertainty_entropy(raw['logits_pred'])
        # entropy_mean = np.mean(entropy)
        # entropy_var = np.var(entropy)
        # visual_img({"image": image, "instances": pseudo_label}, branch="inference")
        # visual_img(
        #     {"image": aug_image, "instances": aug_pseudo_label}, branch="inference"
        # )
        # continue

        del image, aug_image, pseudo_label, aug_pseudo_label
        torch.cuda.empty_cache()

    with open(ORIGINAL_FILE_PATH, "w") as f:
        f.write(json.dumps(dic))

    with open(AUGMENT_FILE_PATH, "w") as f:
        f.write(json.dumps(aug_dic))


def create_pseudo(model, image, pseudo_generator):
    proposals = model.modelTeacher.forward([{"image": image}])

    pseudo_label, num_instance = pseudo_generator.process_pseudo_label(
        proposals,
        0.5,
        "roih",
        "thresholding",
        "inference",
    )

    return pseudo_label, num_instance


def consistency_transform(image):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.RandomErasing(
                p=1.0, scale=(0.005, 0.02), ratio=(0.3, 3.3), value="random"),
            transforms.RandomErasing(
                p=1.0, scale=(0.005, 0.02), ratio=(0.5, 4), value="random"),
        ]
    )
    return transform(image)


def horizontal_flip(image, bbox, location, iteration):
    height, width = image.shape[-2:]
    image = image.flip(-1)
    for i in range(int(iteration)):
        b = bbox[i].cpu()
        x1 = width - b[0]
        x2 = width - b[2]
        location[i][0] = width - location[i][0]
        bbox[i] = torch.Tensor([x2.item(), b[1].item(), x1.item(), b[3].item()])
    return image, Boxes(bbox), location


def record_score_dict(pseudo_label, num_instance, dic, file_name):
    file_name = file_name.split("/")[-1].split(".")[0]
    if num_instance == 0:
        dic[file_name] = []
        box_info = {
            "entropy": np.float(0),  # 엔트로피
            "center": np.float(0),  # 중심에 있는 정도
            "confidence score": np.float(0),  # 분류 정확도
            "score": np.float(0),  # 설정값(cls_n_ctr)으로 분류 정확도 구한 것(T와 비교)
            "pred class": np.int(81),  # 분류 클래스
            "pred box": None,  # bounding box 예측값(l, r, t, b)
        }
        dic[file_name].append(box_info)

        return dic

    entropy = uncertainty_entropy(pseudo_label[0].logits_pred)
    center = pseudo_label[0].centerness.to(device="cpu")
    confidence = pseudo_label[0].cls_confid.to(device="cpu")
    score = pseudo_label[0].scores.to(device="cpu")
    pred_class = pseudo_label[0].gt_classes.to(device="cpu")
    pred_box = pseudo_label[0].gt_boxes.to(device="cpu")

    dic[file_name] = []
    for i in range(len(pseudo_label[0])):
        box_info = {
            "entropy": np.float(entropy),
            "center": np.float(center[i]),
            "confidence score": np.float(confidence[i]),
            "score": np.float(score[i]),
            "pred class": np.int(pred_class[i]),
            "pred box": pred_box.tensor[i].tolist(),
        }
        dic[file_name].append(box_info)

    return dic


def prepare_label_weight(
    dataset_dicts, sup_percent, random_data_seed, random_data_seed_path
):
    num_all = len(dataset_dicts)
    num_label = int(sup_percent / 100.0 * num_all)

    # read from pre-generated data seed
    with PathManager.open(random_data_seed_path, "r") as COCO_sup_file:
        coco_random_idx = json.load(COCO_sup_file)

    labeled_idx = np.array(coco_random_idx[str(sup_percent)][str(random_data_seed)])
    assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

    label_dicts = {}
    labeled_idx = set(labeled_idx)

    for i in range(len(dataset_dicts)):
        if i in labeled_idx:
            for j in dataset_dicts[i]['annotations']:
                if j['category_id'] in label_dicts:
                    label_dicts[j['category_id']] += 1
                else:
                    label_dicts[j['category_id']] = 1

    sorted_keys = sorted(label_dicts.keys())
    sorted_dict = {k: label_dicts[k] for k in sorted_keys}

    values = np.array(list(sorted_dict.values()))
    mean = np.mean(values)
    std = np.std(values)
    normalized_values = (values - mean) / std if std > 0 else np.zeros_like(values)

    # weights = 1 - normalized_values

    # return weights

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--label-file",
        type=str,
        default="../datasets/coco/annotations/instances_train2017.json",
    )
    parser.add_argument(
        "--pick-file",
        type=str,
        default="../"
    )
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
    # parser.add_argument("--model-weights", type=str, default="./model_0109999.pth")
    parser.add_argument("--model-weights", type=str, default="../output/18.9.pth")
    args = parser.parse_args()
    args.eval_only = True
    args.resume = True
    args.num_gpus = 1
    ORIGINAL_FILE_PATH = args.original_file
    AUGMENT_FILE_PATH = args.augmentation_file
    args.config_file = "../configs/fcos/sup1.yaml"  # the config file you used to train this inference model
    # you should config MODEL.WEIGHTS and keep other hyperparameters default(Odd-numbered items are keys, even-numbered items are values)
    args.opts = [
        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
        0.05,
        "MODEL.ROI_HEADS.NMS_THRESH_TEST",
        0.5,
        "TEST.DETECTIONS_PER_IMAGE",
        100,
        "INPUT.FORMAT",
        "BGR",
        "MODEL.WEIGHTS",
        args.model_weights,
    ]
    print("Command Line Args:", args)
    main(args)
