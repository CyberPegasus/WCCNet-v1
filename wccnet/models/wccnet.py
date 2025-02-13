#!/usr/bin/env python
# -*- encoding: utf-8 -*-
 
import torch.nn as nn
from .wccnet_head import WCCNetHead

class WCCNet(nn.Module):

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            assert False, 'backbone is None'
        if head is None:
            assert False, 'head is None'

        self.backbone = backbone
        self.head = head

    def forward(self, rgb, ir, targets=None, illu_label=None):
        
        assert illu_label==None, "illu_label for illumination aware, but deprecated in this repos"
        if illu_label is not None:
            fpn_outs,illu_pred = self.backbone(rgb, ir, illu=True)
        else:
            fpn_outs = self.backbone(rgb, ir)

        if self.training:
            assert targets is not None
            if illu_label is not None:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, illu_loss, num_fg = self.head(
                    xin=fpn_outs, illu_in=illu_pred, labels=targets, illu_label=illu_label, imgs=None
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "illu_loss": illu_loss,
                    "num_fg": num_fg,
                }     
            else:
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    xin=fpn_outs, labels=targets, imgs=None
                )          
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }      

        else:
            outputs = self.head(fpn_outs)

        return outputs
