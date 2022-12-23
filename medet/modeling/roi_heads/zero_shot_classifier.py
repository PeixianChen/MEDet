# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec

class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)
        
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                np.load(zs_weight_path), 
                dtype=torch.float32).permute(1, 0).contiguous() # D x C
        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))], 
            dim=1) # D x (C + 1)
        
        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
        
        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape

        self.seen_id = [0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 21, 22, 23, 24, 26, 28, 29, 30, 33, 37, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 56, 59, 61, 62, 63, 64, 65, 68, 69, 70, 72, 73, 74, 75, 79]
        self.unseen_id = [4, 5, 15, 16, 19, 20, 25, 27, 31, 36, 41, 43, 55, 57, 66, 71, 76]
        self.other_id = [9, 10, 11, 12, 32, 34, 35, 38, 40, 52, 58, 60, 67, 77, 78]
        self.id_dic = {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 7, 6: 8, 7: 13, 8: 14, 9: 17, 10: 18, 11: 21, 12: 22, 13: 23, 14: 24, 15: 26, 16: 28, 17: 29, 18: 30, 19: 33, 20: 37, 21: 39, 22: 42, 23: 44, 24: 45, 25: 46, 26: 47, 27: 48, 28: 49, 29: 50, 30: 51, 31: 53, 32: 54, 33: 56, 34: 59, 35: 61, 36: 62, 37: 63, 38: 64, 39: 65, 40: 68, 41: 69, 42: 70, 43: 72, 44: 73, 45: 74, 46: 75, 47: 79, 48: 4, 49: 5, 50: 15, 51: 16, 52: 19, 53: 20, 54: 25, 55: 27, 56: 31, 57: 36, 58: 41, 59: 43, 60: 55, 61: 57, 62: 66, 63: 71, 64: 76, 65: 9, 66: 10, 67: 11, 68: 12, 69: 32, 70: 34, 71: 35, 72: 38, 73: 40, 74: 52, 75: 58, 76: 60, 77: 67, 78: 77, 79: 78, 80:80}



    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
        }

    def forward(self, x, classifier=None, inference=False, cls_weight=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
        img_embds = x
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous() # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if cls_weight is not None:
            zs_weight = cls_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
            img_embds = x
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x, img_embds
