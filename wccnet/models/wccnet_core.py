#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional
import torch
import torch.nn as nn

from .darknet_IRDWTree import Darknet_IRDWTree,Darknet_DualCNN,Darknet_DualDWT,Darknet_RGBCNN,Darknet_IRDWT,Darknet_Fusion_LateFusion,Darknet_Fusion_MidCat,Darknet_Fusion_DMAF,Darknet_Fusion_InterMA,Darknet_Fusion_EarlyFusion
from .network_blocks import BaseConv
from pytorch_wavelets import DWTForward
    
class WCCNet_backbone(nn.Module):
    """
        WCCNet backbone with FPN neck: DWT-integrated dual-stream Backbone
    """
    def __init__(
        self,
        depth=53,
        in_features=["dark3", "dark4", "dark5"],
        Backbone:str='Darknet_DualStream',
        pre_FDT:Optional[str]=None,
        fpn:str = 'Nr', # Future implemetation list: ['Nr','GA','WA']
        Reverse_RGBIR = False,
        depth_scale:float = 1.0,
        CSA:bool = False,
        dropout2d:float = 0,
        CE:bool = True,
    ):
        super().__init__()
        
        self.pre_FDT = pre_FDT
        self.fpn=fpn
        self.in_features = in_features
        self.Reverse_RGBIR = Reverse_RGBIR

        if self.pre_FDT is not None:
            if self.pre_FDT=='DWT':
                self.preDWT_block = DWTForward(J=1,wave='bior1.3',mode='zero')
                self.irh_upsample = nn.Upsample(scale_factor=2,mode='nearest')
                self.irh_conv = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=5,padding=0,bias=False)
                self.ir_coiff = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1,bias=False)
        
        # NOTE:
        # The variants of our WCCNet is implemented here.
        # The specific used backbone is controlled by the Backbone string variable.
        # The basis WCCNet is implemented in 'Darknet_DualStream'
        if Backbone=='Darknet_DualStream':
            self.backbone = Darknet_IRDWTree(depth=depth, 
                                             IRDWTree=True,
                                             depth_scale=depth_scale,
                                             CSA=CSA,
                                             dropout2d=dropout2d,
                                             CE=CE,
                                            )
        elif Backbone=='Darknet_DualCNN':
            self.backbone = Darknet_DualCNN(depth=depth, 
                                             IRDWTree=False,
                                             depth_scale=depth_scale,
                                             CSA=CSA,
                                             dropout2d=dropout2d,
                                            )
        elif Backbone=='Darknet_DualDWT':
            self.backbone = Darknet_DualDWT(depth=depth, 
                                             IRDWTree=True,
                                             depth_scale=depth_scale,
                                             CSA=CSA,
                                             dropout2d=dropout2d,
                                            )
        elif Backbone=='Darknet_OnlyRGBCNN':
            self.backbone = Darknet_RGBCNN(depth=depth, 
                                             IRDWTree=False,
                                             depth_scale=depth_scale,
                                             CSA=CSA,
                                             dropout2d=dropout2d,
                                            )
        elif Backbone=='Darknet_OnlyIRDWT':
            self.backbone = Darknet_IRDWT(depth=depth, 
                                             IRDWTree=True,
                                             depth_scale=depth_scale,
                                             CSA=CSA,
                                             dropout2d=dropout2d,
                                            )
        elif Backbone=='Darknet_Fusion_EarlyFusion':
            self.backbone = Darknet_Fusion_EarlyFusion(depth=depth, 
                                             depth_scale=depth_scale,
                                             dropout2d=dropout2d,
                                             in_channels=6
                                            )
        elif Backbone=='Darknet_Fusion_LateFusion':
            self.backbone = Darknet_Fusion_LateFusion(depth=depth, 
                                             IRDWTree=True,
                                             depth_scale=depth_scale,
                                             CSA=CSA,
                                             dropout2d=dropout2d,
                                            )
        elif Backbone=='Darknet_Fusion_MidCat':
            self.backbone = Darknet_Fusion_MidCat(depth=depth, 
                                             IRDWTree=True,
                                             depth_scale=depth_scale,
                                             CSA=CSA,
                                             dropout2d=dropout2d,
                                            )       
        elif Backbone=='Darknet_Fusion_DMAF':
            self.backbone = Darknet_Fusion_DMAF(depth=depth, 
                                             IRDWTree=True,
                                             depth_scale=depth_scale,
                                             dropout2d=dropout2d,
                                            )     
        elif Backbone=='Darknet_Fusion_InterMA':
            self.backbone = Darknet_Fusion_InterMA(depth=depth, 
                                             IRDWTree=True,
                                             depth_scale=depth_scale,
                                             dropout2d=dropout2d,
                                            )    
        else:
            assert False, f"not support backbone type {Backbone}"

        out_chs = [128,256,512] 
        if Backbone in ['Darknet_DualDWT','Darknet_OnlyIRDWT']:
            out_chs = [128,256,256]
        out_chs = [int(i*depth_scale) for i in out_chs]
        # out 1
        self.out1_cbl = self._make_cbl(out_chs[2], out_chs[1], 1)
        self.out1 = self._make_embedding([out_chs[1], out_chs[2]], out_chs[1]+out_chs[2])

        # out 2
        self.out2_cbl = self._make_cbl(out_chs[1], out_chs[0], 1)
        if not Backbone in ['Darknet_DualDWT','Darknet_OnlyIRDWT']:
            self.out2 = self._make_embedding([out_chs[0], out_chs[1]], out_chs[1] + out_chs[0])
        else:
            self.out2 = self._make_embedding([out_chs[0], out_chs[1]], out_chs[0] + out_chs[0])

        # upsampler
        assert self.fpn in ['Nr'], print(f'{self.fpn}  is not implemented.')
        self.upsample=nn.Module()
        self.upsample1=nn.Module()
        self.upsample2=nn.Module()
        
        if self.fpn == 'Nr':
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            assert False, print(f'{self.fpn}  is not implemented.')
        
    

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, rgb, ir):
        """
        Args:
            inputs (Tensor): input image.
        Returns:
            Tuple[Tensor]: FPN output features..
        """
        if self.pre_FDT is not None:
            _,irh = self.preDWT_block(torch.sum(ir,dim=1,keepdim=True)) # need ir to be a square img
            irh = irh[0]
            b,c,l,h,w = irh.shape
            irh = irh.view(b,3,h,w) # 2*h = H, H is the height of ir
            irh = self.irh_upsample(irh)
            irh = self.irh_conv(irh)
            ir = torch.cat([ir,irh],dim=1)
            ir = self.ir_coiff(ir)
            
        #  Got multispectral mid feature
        if self.Reverse_RGBIR:
            out_features = self.backbone(ir,rgb)
        else:
            out_features = self.backbone(rgb,ir)
        # x0:512, x1:512, x2:256
        x2,x1,x0 = [out_features[f] for f in self.in_features]

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)
        if self.fpn == 'Nr':
            x1_in = self.upsample(x1_in)
        else:
            assert False
        
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        if self.fpn == 'Nr':
            x2_in = self.upsample(x2_in)
        else:
            assert False

        x2_in = torch.cat([x2_in, x2], 1)

        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, x0)
        return outputs