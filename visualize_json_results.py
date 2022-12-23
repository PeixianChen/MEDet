#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
from torch import gt
import tqdm

from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret

def compute_iou(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    coco_category = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    categories_seen_id = [0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 21, 22, 23, 24, 26, 28, 29, 30, 33, 37, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 56, 59, 61, 62, 63, 64, 65, 68, 69, 70, 72, 73, 74, 75, 79]
    categories_unseen_id = [4, 5, 15, 16, 19, 20, 25, 27, 31, 36, 41, 43, 55, 57, 66, 71, 76]

    unseen2seen, unseen2unseen, seen2seen, seen2unseen = 0, 0, 0, 0
    seenerror, unseenerror = 0, 0
    error_str = []

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])

        pre_boxes = predictions.pred_boxes.tensor.numpy()
        pre_classes = predictions.pred_classes
        gt_boxes = [np.array(d["bbox"]) for d in dic["annotations"]]
        gt_classes = [d["category_id"] for d in dic["annotations"]]

        for pre_ind, pre_box in enumerate(pre_boxes):
            for gt_ind, gt_box in enumerate(gt_boxes):
                gt_box[2] += gt_box[0]
                gt_box[3] += gt_box[1]
                if compute_iou(pre_box, gt_box)>=0.5:
                    if pre_classes[pre_ind] != gt_classes[gt_ind]:
                        if gt_classes[gt_ind] in categories_seen_id:
                            seenerror += 1
                            if pre_classes[pre_ind] in categories_seen_id:
                                seen2seen += 1
                                error_str.append("seen2seen:" + coco_category[gt_classes[gt_ind]] + " " + coco_category[pre_classes[pre_ind]])
                            if pre_classes[pre_ind] in categories_unseen_id:
                                seen2unseen += 1
                                error_str.append("seen2unseen:" + coco_category[gt_classes[gt_ind]] + " " + coco_category[pre_classes[pre_ind]])
                        if gt_classes[gt_ind] in categories_unseen_id: 
                            unseenerror += 1
                            if pre_classes[pre_ind] in categories_seen_id:
                                unseen2seen += 1
                                error_str.append("unseen2seen:" + coco_category[gt_classes[gt_ind]] + " " + coco_category[pre_classes[pre_ind]])
                            if pre_classes[pre_ind] in categories_unseen_id:
                                unseen2unseen += 1
                                error_str.append("unseen2unseen:" + coco_category[gt_classes[gt_ind]] + " " + coco_category[pre_classes[pre_ind]])

        # print(np.min(predictions.scores))
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
    
    print("seenerror:", seenerror)
    print("unseenerror:", unseenerror)
    print("unseen2seen:", unseen2seen)
    print("unseen2unseen:", unseen2unseen)
    print("seen2seen:", seen2seen)
    print("seen2unseen:", seen2unseen)
    print(error_str)
