# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient
from detectron2.modeling.roi_heads.box_head import build_box_head
from urllib3 import connection_from_url

from .detic_fast_rcnn import DeticFastRCNNOutputLayers
from ..debug import debug_second_stage

from torch.cuda.amp import autocast
import random

from .IMRAM import IMRAM, cosine_similarity
from .contrastiveloss import ContrastiveLoss
import torch.nn.functional as F
import os


import glob
all_img_name = glob.glob("/apdcephfs/private_peixianchen/detection/perceptron_v3/datasets/coco/train2017/*.jpg")
all_img_name = [os.path.basename(name) for name in all_img_name]

@ROI_HEADS_REGISTRY.register()
class CustomRes5ROIHeads(Res5ROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        stage_channel_factor = 2 ** 3
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor

        self.with_image_labels = cfg.WITH_IMAGE_LABELS
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.add_image_box = cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX
        self.add_feature_to_prop = cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP
        self.image_box_size = cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE
        self.box_predictor = DeticFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        self.save_debug = cfg.SAVE_DEBUG
        self.save_debug_path = cfg.SAVE_DEBUG_PATH
        if self.save_debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.bgr = (cfg.INPUT.FORMAT == 'BGR')
        
        # IMRAM:
        self.IMRAM = IMRAM(embed_size=512, iteration_step=3, raw_feature_norm="clipped_l2norm", lambda_softmax=11., no_IMRAM_norm=True).cuda()
        self.contrastiveloss = ContrastiveLoss(margin=0.2, max_violation=True)

        self.last_caption_features = None

        self.caption_bbox = {}

        self.bbox_numpy = torch.zeros((1, 4))

        self.caption_txt = open("./caption_txt.txt", "w")
        

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    def forward(self, images, features, proposals, targets=None, iteration=None, 
        ann_type='box', classifier_info=(None,None,None), caption_words_features=None, caps=None, cap_words=None, caption_neg_words_features=None, cap_file_name=None):
        '''
        enable debug and image labels
        classifier_info is shared across the batch
        '''
        batch_size = len(images)
        if self.training:
            if ann_type in ['box']:
                proposals = self.label_and_sample_proposals(
                    proposals, targets)
            else:
                caption_all_proposals = [p[:100] for p in proposals]
                for i in range(len(caption_all_proposals)):
                    if len(caption_all_proposals[i]) < 100:
                        caption_all_proposals[i].proposal_boxes.tensor = torch.cat([caption_all_proposals[i].proposal_boxes.tensor, torch.tensor([[0, 0, 1, 1] for _ in range(100-len(caption_all_proposals[i]))]).cuda()], dim=0)
                        minscore = caption_all_proposals[i].objectness_logits.min()-1
                        caption_all_proposals[i].objectness_logits = torch.cat([caption_all_proposals[i].objectness_logits, torch.zeros(100-len(caption_all_proposals[i])).cuda() + minscore ], dim=0)
                    caption_all_proposals[i].proposal_boxes.clip(caption_all_proposals[i].image_size)
                
                proposals = self.get_top_proposals(proposals)

            
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        # predictions = self.box_predictor(box_features.mean(dim=[2, 3]), classifier_info=classifier_info)

        if self.add_feature_to_prop:
            feats_per_image = box_features.mean(dim=[2, 3]).split(
                [len(p) for p in proposals], dim=0)
            for feat, p in zip(feats_per_image, proposals):
                p.feat = feat


        if self.training:
            predictions = self.box_predictor(box_features.mean(dim=[2, 3]), classifier_info=classifier_info)
            self.coco_emb = predictions[-1]
            # ******Contrastive Learning: RAM LOSS******
            if (ann_type != 'box'): 
                sizes = [boxes.area() for boxes in proposal_boxes]
                ind = [s[:-1].argmax().item() if len(s) > 1 else 0 for s in sizes]
                
                # add image_size proposals
                for ind, p in enumerate(caption_all_proposals):
                    p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
                    caption_all_proposals[ind] = self._add_image_box(p)

                caption_proposal_boxes = [x.proposal_boxes for x in caption_all_proposals]
                caption_box_features = self._shared_roi_transform(
                    [features[f] for f in self.in_features], caption_proposal_boxes
                )
                caption_predictions = self.box_predictor(
                    caption_box_features.mean(dim=[2, 3]),
                    classifier_info=classifier_info)
                caption_instance, caption_words_features_ = self.noise_removal(iteration, caption_all_proposals, caption_predictions, cap_words, caption_words_features, features, classifier_info, cap_file_name=cap_file_name, TOPK=3)
               
            del features
            if (ann_type != 'box'):
                image_labels = [x._pos_category_ids for x in targets]
                losses = self.box_predictor.image_label_losses(
                    predictions, proposals, image_labels,
                    classifier_info=classifier_info,
                    ann_type=ann_type)
                
                # ram loss:
                caption_features2ram = caption_words_features_
                caption_features2ram = [cap for cap in caption_features2ram if cap != []]
                cam_lens = [len(cap_emb) for cap_emb in caption_features2ram]
                ram_score = self.IMRAM(caption_instance, caption_features2ram, cam_lens)
                losses['RAM_loss'], diagonal = self.contrastiveloss(ram_score)
                losses['RAM_loss'] = losses['RAM_loss']  / batch_size
            else:
                if self.last_caption_features is not None:
                    predictions = self.box_predictor(box_features.mean(dim=[2, 3]), classifier_info=classifier_info, cls_weight=self.last_caption_features.T)
                losses = self.box_predictor.losses(
                    (predictions[0], predictions[1]), proposals)
                if self.with_image_labels:
                    assert 'image_loss' not in losses
                    losses['image_loss'] = predictions[0].new_zeros([1])[0]

            if self.save_debug:
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                if ann_type != 'box':
                    image_labels = [x._pos_category_ids for x in targets]
                else:
                    image_labels = [[] for x in targets]
                debug_second_stage(
                    [denormalizer(x.clone()) for x in images],
                    targets, proposals=proposals,
                    save_debug=self.save_debug,
                    debug_show_name=self.debug_show_name,
                    vis_thresh=self.vis_thresh,
                    image_labels=image_labels,
                    save_debug_path=self.save_debug_path,
                    bgr=self.bgr)
            losses['image_loss'] = predictions[0].new_zeros([1])[0]
            losses['loss_cls'] = predictions[0].new_zeros([1])[0]
            losses['loss_box_reg'] = predictions[0].new_zeros([1])[0]
            losses['RAM_loss'] = predictions[0].new_zeros([1])[0]
            return proposals, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            if self.save_debug:
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                debug_second_stage(
                    [denormalizer(x.clone()) for x in images],
                    pred_instances, proposals=proposals,
                    save_debug=self.save_debug,
                    debug_show_name=self.debug_show_name,
                    vis_thresh=self.vis_thresh,
                    save_debug_path=self.save_debug_path,
                    bgr=self.bgr)
            # using offline class-wise adjustment (OCA)
            beta_npy_path = './datasets/coco_cls_beta.npy'
            cls_beta = predictions[0].new_tensor(np.load(beta_npy_path))
            max_cls_beta = max(cls_beta).clone()
            cls_beta[cls_beta==0] = max_cls_beta * 5
            cal_factor = predictions[0].new_ones(predictions[0].shape[1])
            cal_factor[:-1] = torch.pow(cls_beta * 8, 0.4)
            # ii) rescale the predictions[0] with cal_factor
            predictions_Cal = torch.exp(predictions[0]) / cal_factor.reshape(1,-1)
            predictions_Cal_normed = predictions_Cal / torch.sum(predictions_Cal, 1).reshape(-1,1)
            predictions_new = (predictions_Cal_normed, predictions[1], predictions[2], predictions[3])
            pred_instances, _ = self.box_predictor.inference(predictions_new, proposals)
            return pred_instances, {}

    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps))

    def fragment_mergence(self, box1, box2):
        x0 = min(box1[0], box2[0])
        y0 = min(box1[1], box2[1])
        x1 = max(box1[2], box2[2])
        y1 = max(box1[3], box2[3])
        return torch.tensor([x0, y0, x1, y1])[None, ...].cuda()
    

    def get_iou(self, pred_box, gt_box):
        ixmin = max(pred_box[0], gt_box[0])
        ixmax = min(pred_box[2], gt_box[2])
        iymin = max(pred_box[1], gt_box[1])
        iymax = min(pred_box[3], gt_box[3])
        iw = np.maximum(ixmax-ixmin+1., 0.)
        ih = np.maximum(iymax-iymin+1., 0.)
        inters = iw*ih
        uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
            (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
            inters)
        iou = inters / uni
        return iou


    def noise_removal(self, iteration, proposals, predictions, caption_words, caption_features, all_features, classifier_info, cap_file_name=None, TOPK=3):
        # choose RPN top-100
        image_embding = predictions[-2].view(-1, 101, 512)
        all_add_proposals = []
        img_box_num = []
        neg_instance_embedding = []
        caption_f = []
        caption_word_f = []
        self.last_caption_features = self.coco_emb.T
        for ind, cap_feature in enumerate(caption_features):
            if cap_feature == []:
                continue

            self.coco_seen_emb = self.coco_emb.T
            self.last_caption_features = torch.cat([self.last_caption_features, cap_feature], dim=0)

            # similarity entropy:
            SE_cap_feature = torch.cat([self.coco_seen_emb, cap_feature], dim=0)
            rpnclipscores = torch.nn.Softmax()(torch.mm(image_embding[ind], SE_cap_feature.T) * torch.nn.Sigmoid()(proposals[ind].objectness_logits)[..., None]) + 1e-40
            simentropy = -1 * torch.sum(torch.log(rpnclipscores) * rpnclipscores, dim=1)
            sim_index = torch.where(simentropy <= simentropy[-1])

            if sim_index[0].shape[0] > TOPK:
                sim_index = sim_index[0]
                k_num = TOPK
            else:
                sim_index = sim_index[0]
                k_num = sim_index.shape[0]
            image_embding_ = image_embding[ind][sim_index]
            objectness_logits_ = proposals[ind].objectness_logits[sim_index]
            proposals_box = proposals[ind].proposal_boxes.tensor[sim_index]

            # choose TOP-3 for every concept:
            rpnclipscores = torch.nn.Softmax()(torch.mm(image_embding_, cap_feature.T)) * torch.nn.Sigmoid()(objectness_logits_)[..., None]
            topk_indexs = torch.topk(rpnclipscores, k_num, dim=0, largest=True)[1].T

            add_boxs = []
            words_feature = []
            words_word = []
            for indw, topk in enumerate(topk_indexs):
                if objectness_logits_.shape[0]-1 == topk[0]:
                    continue
                words_feature.append(cap_feature[indw].unsqueeze(0))
                img_boxes = []
                maxscore_proposals = proposals_box[topk[0].item()][None, ...]
                img_boxes.append(maxscore_proposals)
                for k in topk[1:]:
                    hadmerge=False
                    for indx in range(len(img_boxes)):
                        if self.get_iou(img_boxes[indx][0].cpu().numpy(), proposals_box[k].cpu().numpy()) > 0.6:
                            hadmerge=True
                            img_boxes[indx] = self.fragment_mergence(img_boxes[indx][0], proposals_box[k])
                            break
                    if not hadmerge:
                        img_boxes.append(proposals_box[k][None, ...])
                words_word.extend([caption_words[ind][indw] for _ in range(len(img_boxes))])
                add_boxs.extend(img_boxes)

            if words_feature != []:
                caption_f.append(torch.cat(words_feature, dim=0))
                caption_word_f.append(words_word)
            else:
                caption_f.append([])
                caption_word_f.append([])

            if add_boxs != []:
                add_boxs = torch.cat(add_boxs, dim=0)
                add_boxs = torch.unique(add_boxs, dim=0)
                img_box_num.append(add_boxs.shape[0])
                add_proposals = Instances(proposals[ind].image_size)
                add_proposals.proposal_boxes = Boxes(add_boxs)
                all_add_proposals.append(add_proposals)


        

        add_proposal_boxes = [x.proposal_boxes for x in all_add_proposals]


        features = [all_features[f] for f in self.in_features]
        all_features = torch.cat([f.unsqueeze(0) for index, f in enumerate(features[0]) if caption_features[index]!=[]], dim=0)
        all_features = torch.cat([f.unsqueeze(0) for index, f in enumerate(all_features) if caption_f[index]!=[]], dim=0)
        add_box_features = self._shared_roi_transform([all_features], add_proposal_boxes)
        add_predictions = self.box_predictor(
            add_box_features.mean(dim=[2, 3]),
            classifier_info=classifier_info)
        instance_embedding = list(torch.split(add_predictions[-2], img_box_num, dim=0))
        ins_num = max(img_box_num)
        for ind, instance in enumerate(instance_embedding):
            instance = [ins.unsqueeze(0) for ins in instance]
            for i in range(ins_num):
                if len(instance) >= ins_num:
                    continue
                instance.append(instance[i])
            instance_embedding[ind] = torch.cat(instance[:ins_num], dim=0).unsqueeze(0)
        caption_f = [caps for caps in caption_f if caps != []]
        return torch.cat(instance_embedding, dim=0), caption_f

    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        # proposals = [p[:self.ws_num_props] for p in proposals]
        proposals = [p[:32] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals
    
    
    def inpacedmaxsize(self, proposals, boxes, confidences):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.tensor[0] = boxes[i][confidences[i].argmax()]
        return proposals
    
    def _add_image_box(self, p, use_score=False):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        if self.image_box_size < 1.0:
            f = self.image_box_size
            image_box.proposal_boxes = Boxes(
                p.proposal_boxes.tensor.new_tensor(
                    [w * (1. - f) / 2., 
                        h * (1. - f) / 2.,
                        w * (1. - (1. - f) / 2.), 
                        h * (1. - (1. - f) / 2.)]
                    ).view(n, 4))
        else:
            image_box.proposal_boxes = Boxes(
                p.proposal_boxes.tensor.new_tensor(
                    [0, 0, w, h]).view(n, 4))
        if use_score:
            image_box.scores = \
                p.objectness_logits.new_ones(n)
            image_box.pred_classes = \
                p.objectness_logits.new_zeros(n, dtype=torch.long) 
            image_box.objectness_logits = \
                p.objectness_logits.new_ones(n) 
        else:
            image_box.objectness_logits = \
                p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])

