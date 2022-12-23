#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from itertools import chain
import cv2
import tqdm
import sys

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch


from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config

sys.path.insert(0, 'third_party/Deformable-DETR')
from detic.config import add_detic_config
from detic.data.custom_build_augmentation import build_custom_augmentation
from detic.data.custom_dataset_dataloader import  build_custom_train_loader
from detic.modeling.utils import reset_cls_test



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="detic")
    return cfg

def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 1.0
    if args.source == "dataloader":
        train_data_loader = build_custom_train_loader(cfg)
        for i, batch in enumerate(train_data_loader):
            for per_image in batch:
                # Pytorch tensor is in (C, H, W) format
                img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                import pdb; pdb.set_trace()
                target_fields = per_image["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                output(vis, str(per_image["image_id"]) + ".jpg")
    else:
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
        if cfg.MODEL.KEYPOINT_ON:
            dicts = filter_images_with_few_keypoints(dicts, 1)
        for dic in tqdm.tqdm(dicts):
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))