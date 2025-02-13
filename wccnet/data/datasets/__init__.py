#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# WCCNet is extended from YOLOX

from .coco import COCODataset
from .coco_classes import COCO_CLASSES, KAIST_CLASSES, FLIR_CLASSES, LLVIP_CLASSES
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .mosaicdetection_multispectral import MosaicDetection_Multispectral
from .voc import VOCDetection
