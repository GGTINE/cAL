import copy
import cv2
import numpy as np

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.events import get_event_storage


coco_class = {
    0: "__background__",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush",
}


def visual_img(data):
    for x in data:
        image = x["image"]
        image = convert_image_to_rgb(image.permute(1, 2, 0), "RGB")
        image2 = copy.deepcopy(image)
        for i in range(len(x["instances_class"])):
            classes = x["instances_class"][i].gt_classes.tolist()[0]
            boxes = x["instances_class"][i].gt_boxes.tensor.tolist()[0]
            certers = x["instances_class"][i].gt_boxes.get_centers().tolist()[0]
            cv2.rectangle(
                image,
                (int(boxes[0]), int(boxes[1])),
                (int(boxes[2]), int(boxes[3])),
                (0, 255, 0),
                1,
            )
            cv2.putText(
                image,
                str(coco_class[classes + 1]),
                (int(boxes[0]), int(boxes[1] + 20)),
                cv2.FONT_ITALIC,
                1,
                (255, 0, 0),
                3,
            )
            cv2.circle(image, (int(certers[0]), int(certers[1])), 5, (0, 0, 255))

        for i in range(len(x["instances_reg"])):
            classes = x["instances_reg"][i].gt_classes.tolist()[0]
            boxes = x["instances_reg"][i].gt_boxes.tensor.tolist()[0]
            certers = x["instances_reg"][i].gt_boxes.get_centers().tolist()[0]
            cv2.rectangle(
                image2,
                (int(boxes[0]), int(boxes[1])),
                (int(boxes[2]), int(boxes[3])),
                (0, 255, 0),
                1,
            )
            cv2.putText(
                image2,
                str(coco_class[classes + 1]),
                (int(boxes[0]), int(boxes[1] + 20)),
                cv2.FONT_ITALIC,
                1,
                (255, 0, 0),
                3,
            )
            cv2.circle(image2, (int(certers[0]), int(certers[1])), 5, (0, 0, 255))

        image = np.hstack((image, image2))
        cv2.imwrite("../../result.png", image)
        cv2.imshow("", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_training(batched_inputs, proposals, branch):
    """
    A function used to visualize images and proposals. It shows ground truth
    bounding boxes on the original image and up to 20 top-scoring predicted
    object proposals on the original image. Users can implement different
    visualization functions for different models.

    Args:
        batched_inputs (list): a list that contains input to the model.
        proposals (list): a list that contains predicted proposals. Both
            batched_inputs and proposals should have the same length.
    """
    from detectron2.utils.visualizer import Visualizer

    storage = get_event_storage()
    max_vis_prop = 20

    for input, prop in zip(batched_inputs, proposals):
        if branch == "labeled":
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), "BGR")
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes.to("cpu"))
            anno_img = v_gt.get_image()
            box_size = min(len(prop.pred_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                branch + " | Left: GT bounding boxes;      Right: Predicted proposals"
            )
        elif branch == "unlabeled":
            img_list = []
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), "BGR")

            # classification pseudo-set
            if "instances_class" in input:
                v_gt = Visualizer(img, None)
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances_class"].gt_boxes.to("cpu")
                )
                anno_img = v_gt.get_image()
                img_list.append(anno_img)

            # regression pseudo-set
            if "instances_reg" in input:
                v_gt2 = Visualizer(img, None)
                v_gt2 = v_gt2.overlay_instances(
                    boxes=input["instances_reg"].gt_boxes.to("cpu")
                )
                anno_reg_img = v_gt2.get_image()
                img_list.append(anno_reg_img)

            box_size = min(len(prop.pred_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.pred_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            img_list.append(prop_img)

            vis_img = np.concatenate(tuple(img_list), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)

            vis_name = (
                branch
                + " | Left: Pseudo-Cls; Center: Pseudo-Reg; Right: Predicted proposals"
            )
        else:
            break
        storage.put_image(vis_name, vis_img)
        break  # only visualize one image in a batch
