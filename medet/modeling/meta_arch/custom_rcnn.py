# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, nonzero
import json
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
import detectron2.utils.comm as comm

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb

from torch.cuda.amp import autocast
from ..text.text_encoder import build_text_encoder
from ..utils import load_class_freq, get_fed_loss_inds
import random

from ..text.cross_encoder import CROSSTRANS
from torch.nn import functional as F


@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        with_image_labels = False,
        dataset_loss_weight = [],
        fp16 = False,
        sync_caption_batch = False,
        roi_head_name = '',
        cap_batch_ratio = 4,
        with_caption = False,
        dynamic_classifier = False,
        **kwargs):
        """
        """
        self.with_image_labels = with_image_labels
        self.dataset_loss_weight = dataset_loss_weight
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = False

        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(pretrain=True)
            for v in self.text_encoder.parameters():
                v.requires_grad = False

        self.categories_seen_id = [0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 21, 22, 23, 24, 26, 28, 29, 30, 33, 37, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 56, 59, 61, 62, 63, 64, 65, 68, 69, 70, 72, 73, 74, 75, 79, 80]
        self.categories_unseen_id = [4, 5, 15, 16, 19, 20, 25, 27, 31, 36, 41, 43, 55, 57, 66, 71, 76]
        self.seen_dict = {}
        for inds, ids in enumerate(self.categories_seen_id):
            self.seen_dict[ids] = inds

        self.coco_category = ['a person', 'a bicycle', 'a car', 'a motorcycle', 'a airplane', 'a bus', 'a train', 'a truck', 'a boat', 'a traffic light', 'a fire hydrant', 'a stop sign', 'a parking meter', 'a bench', 'a bird', 'a cat', 'a dog', 'a horse', 'a sheep', 'a cow', 'a elephant', 'a bear', 'a zebra', 'a giraffe', 'a backpack', 'a umbrella', 'a handbag', 'a tie', 'a suitcase', 'a frisbee', 'a skis', 'a snowboard', 'a sports ball', 'a kite', 'a baseball bat', 'a baseball glove', 'a skateboard', 'a surfboard', 'a tennis racket', 'a bottle', 'a wine glass', 'a cup', 'a fork', 'a knife', 'a spoon', 'a bowl', 'a banana', 'a apple', 'a sandwich', 'a orange', 'a broccoli', 'a carrot', 'a hot dog', 'a pizza', 'a donut', 'a cake', 'a chair', 'a couch', 'a potted plant', 'a bed', 'a dining table', 'a toilet', 'a tv', 'a laptop', 'a mouse', 'a remote', 'a keyboard', 'a cell phone', 'a microwave', 'a oven', 'a toaster', 'a sink', 'a refrigerator', 'a book', 'a clock', 'a vase', 'a scissors', 'a teddy bear', 'a hair drier', 'a toothbrush']
        self.coco_weight = torch.tensor(np.load("./datasets/metadata/coco_v2_clip_a+cname.npy"), dtype=torch.float32).contiguous().cuda()

        self.trans_bfrpn, self.labels_seen = None, None

        # concept augmentation:

        self.selmode = 'allwordsaug'      
        self.layer = 'p4'  
        self.trans_bfrpn = CROSSTRANS(in_chans=1024, embed_dim=512, key_dim=64)
        self.labels_seen_adda = [self.coco_category[idx] for idx in self.categories_seen_id[:-1]]
        self.labels_seen_idxs = self.categories_seen_id[:-1] 
        self.coco_seen_emb = torch.cat([self.coco_weight[index].unsqueeze(0) for index in self.labels_seen_idxs], dim=0)
        self.wordscan = None


        with open('./datasets/coco/zero-shot/instances_val2017_all_2.json', 'r') as fin:
            datatmp = json.load(fin)
        self.labels_all = [datatmp['categories'][idx]['name']  for idx in range(len(datatmp['categories']))]
        self.labels_all_ids = []
        for idx in range(len(datatmp['categories'])):
            nametmp = 'a ' + datatmp['categories'][idx]['name']
            for iid in range(len(self.coco_category)):
                if nametmp == self.coco_category[iid]:
                    self.labels_all_ids.append(iid)
                    break


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'dataset_loss_weight': cfg.MODEL.DATASET_LOSS_WEIGHT,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
        })
        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS
        return ret


    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        cls_features, cls_inds, caption_features = None, None, None
        classifier_info = cls_features, cls_inds, caption_features

        wordscan = []
        if self.selmode == 'allwordsaug':
            if len(features) > 1:
                wordscan = self.lvis_name_adda
            else:
                wordscan = self.coco_category


        words_feature = self.text_encoder(wordscan).float()
        if self.selmode != 'allwordsaug':
            features = self.trans_bfrpn(features, words_feature)
        else:
            if len(features) > 1:
                if self.layer == 'cmb':
                    _, _, H, W = features['p4'].shape
                    tmp1 = F.interpolate(features['p3'], (H, W))
                    tmp2 = F.interpolate(features['p5'], (H, W))
                    tmp3 = F.interpolate(features['p6'], (H, W))
                    tmp4 = F.interpolate(features['p7'], (H, W))
                    cmbfea = torch.cat((tmp1, features['p4'], tmp2, tmp3, tmp4), 1)
                    words_feature_aug = self.trans_bfrpn(cmbfea, words_feature.detach())
                else:
                    words_feature_aug = self.trans_bfrpn(features[self.layer], words_feature.detach())
            else:
                words_feature_aug = self.trans_bfrpn(features['res4'], words_feature.detach())
            zs_weight = torch.cat(
                [words_feature_aug, words_feature_aug.new_zeros((1, words_feature_aug.shape[1]))],
                    dim=0)
            cls_features = zs_weight
            classifier_info = cls_features, cls_inds, caption_features


        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, classifier_info=classifier_info)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], iteration=0):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        ann_type = 'box'
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        for index in range(len(gt_instances)):
            for gt in range(gt_instances[index].gt_classes.shape[0]):
                gt_instances[index].gt_classes[gt] = self.seen_dict[gt_instances[index].gt_classes[gt].item()]
        
        if self.with_image_labels:
            for inst, x in zip(gt_instances, batched_inputs):
                inst._ann_type = x['ann_type']
                inst._pos_category_ids = x['pos_category_ids']
            ann_types = [x['ann_type'] for x in batched_inputs]
            assert len(set(ann_types)) == 1
            ann_type = ann_types[0]
            if ann_type in ['prop', 'proptag']:
                for t in gt_instances:
                    t.gt_classes *= 0
        
        if self.fp16: # TODO (zhouxy): improve
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)

        cls_features, cls_inds, caption_features = None, None, None
        caption_words_features, caps, cap_words = None, None, None
        caption_neg_words_features = None
        words_feature_aug = None

        # if self.use_trans_bfrpn:
        ####
        wordscan = []
        if self.selmode ==  'seenwords':
            wordscan = self.labels_seen_adda
        elif self.selmode == 'allwords':
            wordscan = self.coco_category
        elif self.selmode == 'captiontoken':
            wordscan = ['a photo of detection objects']
        elif self.selmode == 'allwordsaug':
            if len(features) > 1:
                wordscan = self.lvis_seen_name_adda
            else:
                wordscan = self.labels_seen_adda  #self.coco_category
        
        caption_features = None
        if ((self.with_caption) and ('caption' in ann_type)):
            if self.selmode == 'captiontoken':
                inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                            for x in batched_inputs]
                caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
                caption_features = self.text_encoder(caps, elewise=True)
            else:
                addcon = []
                for x in batched_inputs:
                    if (('object' in x)  and (len(x['object']) > 0)):
                        #wordscan = wordscan + x['object']
                        for objname in x['object']:
                            if (('a '+objname) not in wordscan):
                                addcon.append('a ' + objname)
                wordscan = wordscan + addcon
                self.wordscan = wordscan

        if self.wordscan is not None:
            wordscan = self.wordscan

        words_feature = None
        if self.selmode == 'captiontoken':
            if ((self.with_caption) and ('caption' in ann_type)):
                words_feature = caption_features
            else:
                B = len(batched_inputs)
                wordscan = ['a photo of detection objects' for iid in range(B)]
                words_feature = self.text_encoder(wordscan, elewise=True)
        else:
            words_feature = self.text_encoder(wordscan).float()

        if self.selmode != 'allwordsaug':
            features = self.trans_bfrpn(features, words_feature)
        else:
            if len(features) > 1:
                if self.layer == 'cmb':
                    _, _, H, W = features['p4'].shape
                    tmp1 = F.interpolate(features['p3'], (H, W))
                    tmp2 = F.interpolate(features['p5'], (H, W))
                    tmp3 = F.interpolate(features['p6'], (H, W))
                    tmp4 = F.interpolate(features['p7'], (H, W))
                    cmbfea = torch.cat((tmp1, features['p4'], tmp2, tmp3, tmp4), 1)
                    words_feature_aug = self.trans_bfrpn(cmbfea, words_feature.detach())
                else:
                    words_feature_aug = self.trans_bfrpn(features[self.layer], words_feature.detach())
                zs_weight = self.lvis_weight.clone().detach()
                for iid in range(len(self.lvis_seen_id)):
                    zs_weight[self.lvis_seen_id[iid], :] = words_feature_aug[iid, :].clone()
                zs_weight = torch.cat(
                    [zs_weight, zs_weight.new_zeros((1, words_feature_aug.shape[1]))], dim=0)
                cls_features = zs_weight
            else:
                words_feature_aug = self.trans_bfrpn(features['res4'], words_feature.detach())
                #zs_weight = torch.cat(
                #    [words_feature_aug, words_feature_aug.new_zeros((1, words_feature_aug.shape[1]))],
                #     dim=0)
                zs_weight = self.coco_seen_emb.clone().detach()
                for iid in range(len(self.labels_seen_idxs)):
                    zs_weight[iid, :] = words_feature_aug[iid, :].clone()
                zs_weight = torch.cat(
                    [zs_weight, zs_weight.new_zeros((1, words_feature_aug.shape[1]))], dim=0)
                cls_features = zs_weight

        if self.with_caption and 'caption' in ann_type:
            inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                for x in batched_inputs]
            caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
            caption_features = self.text_encoder(caps).float()
            cap_words = [x['object'] for x in batched_inputs]
            caption_words_features = []
            for words in cap_words:
                if words == []:
                    words_feature = []
                else:
                    words_feature = self.text_encoder(words).float()
                caption_words_features.append(words_feature)

            # if ((self.use_trans_bfrpn) and (self.selmode == 'allwordsaug')):
            if self.selmode == 'allwordsaug':
                #####v2
                initc = len(self.labels_seen_idxs) if len(features)==1 else len(self.lvis_seen_name_adda)
                checkwordscan = None
                if len(features) > 1:
                    checkwordscan = self.lvis_seen_name_adda
                else:
                    checkwordscan = self.labels_seen_adda

                for bi in range(len(cap_words)):
                    if len(cap_words[bi]) == 0:
                        continue
                    for ind, w in enumerate(cap_words[bi]):
                        if ('a '+w) not in checkwordscan:
                            caption_words_features[bi][ind] = words_feature_aug[initc].clone()
                            initc += 1

                
        if self.sync_caption_batch:
            caption_features = self._sync_caption_features(
                caption_features, ann_type, len(batched_inputs))

        if self.dynamic_classifier and ann_type != 'caption':
            cls_inds = self._sample_cls_inds(gt_instances, ann_type) # inds, inv_inds
            ind_with_bg = cls_inds[0].tolist() + [-1]
            cls_features = self.roi_heads.box_predictor[
                0].cls_score.zs_weight[:, ind_with_bg].permute(1, 0).contiguous()

        classifier_info = cls_features, cls_inds, caption_features
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)

        if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances)
        else:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances,
                iteration=iteration, ann_type=ann_type, classifier_info=classifier_info, caption_words_features=caption_words_features, caps=caps, cap_words=cap_words)#, caption_neg_words_features=caption_neg_words_features)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        if self.with_image_labels:
            if ann_type in ['box', 'prop', 'proptag']:
                losses.update(proposal_losses)
            else: # ignore proposal loss for non-bbox data
                losses.update({k: v * 0 for k, v in proposal_losses.items()})
        else:
            losses.update(proposal_losses)
        if len(self.dataset_loss_weight) > 0:
            dataset_sources = [x['dataset_source'] for x in batched_inputs]
            assert len(set(dataset_sources)) == 1
            dataset_source = dataset_sources[0]
            for k in losses:
                losses[k] *= self.dataset_loss_weight[dataset_source]
        
        if self.return_proposal:
            return proposals, losses
        else:
            return losses


    def _sync_caption_features(self, caption_features, ann_type, BS):
        has_caption_feature = (caption_features is not None)
        BS = (BS * self.cap_batch_ratio) if (ann_type == 'box') else BS
        rank = torch.full(
            (BS, 1), comm.get_rank(), dtype=torch.float32, 
            device=self.device)
        if not has_caption_feature:
            caption_features = rank.new_zeros((BS, 512))
        caption_features = torch.cat([caption_features, rank], dim=1)
        global_caption_features = comm.all_gather(caption_features)
        caption_features = torch.cat(
            [x.to(self.device) for x in global_caption_features], dim=0) \
                if has_caption_feature else None # (NB) x (D + 1)
        return caption_features


    def _sample_cls_inds(self, gt_instances, ann_type='box'):
        if ann_type == 'box':
            gt_classes = torch.cat(
                [x.gt_classes for x in gt_instances])
            C = len(self.freq_weight)
            freq_weight = self.freq_weight
        else:
            gt_classes = torch.cat(
                [torch.tensor(
                    x._pos_category_ids, 
                    dtype=torch.long, device=x.gt_classes.device) \
                    for x in gt_instances])
            C = self.num_classes
            freq_weight = None
        assert gt_classes.max() < C, '{} {}'.format(gt_classes.max(), C)
        inds = get_fed_loss_inds(
            gt_classes, self.num_sample_cats, C, 
            weight=freq_weight)
        cls_id_map = gt_classes.new_full(
            (self.num_classes + 1,), len(inds))
        cls_id_map[inds] = torch.arange(len(inds), device=cls_id_map.device)
        return inds, cls_id_map
