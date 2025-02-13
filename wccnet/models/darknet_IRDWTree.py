from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, ResLayer, SPPBottleneck, padding_for_fpn, ir2gray
from .wavelet_transform import DWT
from .CMRF import CMRFusion,CSACMRFusion
from .other_fusion_modes import DMAF,InterMA

class Darknet_IRDWTree(nn.Module):
    # implementation for DWT-integrated Dual-stream Backbone

    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        IRDWTree:bool=False,
        CSA: bool=False,
        depth_scale:float = 1,
        dropout2d:float = 0,
        CE:bool=True,
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        self.IRDWTree = IRDWTree
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)

        self.stem = nn.Sequential( # backbone.backbone.stem
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu",not_pad=self.paddingCropped),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,not_pad=self.paddingCropped),
        )
        if CE:
            self.embedding1 = BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1)
        else:
            self.embedding1 = nn.Conv2d(3, stem_out_channels, 1)   

        if not CSA:
            self.CMRF1 = CMRFusion(stem_out_channels,stem_out_channels*2)
        else:
            self.CMRF1 = CSACMRFusion(stem_out_channels,stem_out_channels*2)

        in_channels = stem_out_channels * 2  # 64
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential( # backbone.backbone.dark2
            *self.make_group_layer(in_channels, num_blocks[0], stride=2,not_pad=self.paddingCropped)
        )
        if CE:
            self.embedding2 = nn.Sequential(
                BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
                BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
                ) # 16
        else:
            self.embedding2 = nn.Conv2d(3, in_channels, 1)

        if not CSA:
            self.CMRF2 = CMRFusion(in_channels,in_channels*2)
        else:
            self.CMRF2 = CSACMRFusion(in_channels,in_channels*2)

        in_channels *= 2  # 128
        self.dark3 = nn.Sequential( # backbone.backbone.dark3
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        if CE:
            self.embedding3 = nn.Sequential(
                BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
                BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
                ) # 16
        else:
            self.embedding3 =nn.Conv2d(3, in_channels, 1)

        if not CSA:
            self.CMRF3 = CMRFusion(in_channels,in_channels*2)
        else:
            self.CMRF3 = CSACMRFusion(in_channels,in_channels*2)

        in_channels *= 2  # 256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )

        in_channels *= 2  # 512
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )

        # IR wavelet tree
        self.Wavelet = DWT(wave='bior1.3')
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir, illu_aware:bool = False, debug_vis:bool=True):
        outputs = {}
        if self.IRDWTree:
            ir = ir2gray(ir) # b,c,h,w -> b,1,h,w
        
        # level 1
        ll,lh = self.Wavelet(ir,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding1(ir)
        x = self.stem(x)
        outputs["stem"] = x
        if not debug_vis:
            x = self.CMRF1(x,ir)
        else:
            x, y = self.CMRF1(x,ir,debug_vis)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 2
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding2(ir)
        x = self.dark2(x)
        outputs["dark2"] = x
        if not debug_vis:
            x = self.CMRF2(x,ir)
        else:
            x, y = self.CMRF2(x,ir,debug_vis)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 3
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding3(ir)
        x = self.dark3(x)
        outputs["dark3"] = x
        if not debug_vis:
            x = self.CMRF3(x,ir)
        else:
            x, y = self.CMRF3(x,ir,debug_vis)   
                 
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 4
        x = self.dark4(x)
        outputs["dark4"] = x
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 5
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}

class Darknet_DualCNN(nn.Module):
    # implementation for dilated Cross Stage Concat
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        IRDWTree:bool=False,
        Illu_scale:bool=False,
        CSA: bool=False,
        depth_scale:float = 1,
        dropout2d:float = 0
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        self.IRDWTree = IRDWTree
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)

        self.stem = nn.Sequential( # backbone.backbone.stem
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu",not_pad=self.paddingCropped),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,not_pad=self.paddingCropped),
        )
        self.stem_2 = nn.Sequential( # backbone.backbone.stem
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu",not_pad=self.paddingCropped),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,not_pad=self.paddingCropped),
        )
        # self.embedding1 = nn.Sequential(
        #     BaseConv(3, stem_out_channels//2, ksize=3, act="lrelu", stride=1),
        #     BaseConv(stem_out_channels//2, stem_out_channels, ksize=3, act="lrelu", stride=1),
        #     ) # 16
        if not CSA:
            self.CMRF1 = CMRFusion(stem_out_channels*2,stem_out_channels*2)
        else:
            self.CMRF1 = CSACMRFusion(stem_out_channels*2,stem_out_channels*2)

        in_channels = stem_out_channels * 2  # 64
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential( # backbone.backbone.dark2
            *self.make_group_layer(in_channels, num_blocks[0], stride=2,not_pad=self.paddingCropped)
        )
        self.dark2_2 = nn.Sequential( # backbone.backbone.dark2
            *self.make_group_layer(in_channels, num_blocks[0], stride=2,not_pad=self.paddingCropped)
        )
        if not CSA:
            self.CMRF2 = CMRFusion(in_channels*2,in_channels*2)
        else:
            self.CMRF2 = CSACMRFusion(in_channels*2,in_channels*2)

        in_channels *= 2  # 128
        self.dark3 = nn.Sequential( # backbone.backbone.dark3
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        self.dark3_2 = nn.Sequential( # backbone.backbone.dark3
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        if not CSA:
            self.CMRF3 = CMRFusion(in_channels*2,in_channels*2)
        else:
            self.CMRF3 = CSACMRFusion(in_channels*2,in_channels*2)

        in_channels *= 2  # 256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        self.dark4_2 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        if not CSA:
            self.CMRF4 = CMRFusion(in_channels*2,in_channels*2)
        else:
            self.CMRF4 = CSACMRFusion(in_channels*2,in_channels*2)

        in_channels *= 2  # 512
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )
        self.dark5_2 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            ) # SPP的输出channel数又变成in_channels
        if not CSA:
            self.CMRF5 = CMRFusion(in_channels,in_channels)
        else:
            self.CMRF5 = CSACMRFusion(in_channels,in_channels)
        
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir, illu_aware = False):
        outputs = {}
        x = self.stem(x)
        ir = self.stem_2(ir)
        outputs["stem"] = x
        x = self.CMRF1(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 2
        ir = self.dark2_2(ir)
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.CMRF2(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 3
        ir = self.dark3_2(ir)
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.CMRF3(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 4
        ir = self.dark4_2(ir)
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.CMRF4(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)
        # x = self.CMRF4(x,ir)

        # level 5
        ir = self.dark5_2(ir)
        x = self.dark5(x)
        outputs["dark5"] = x
        x = self.CMRF5(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
class Darknet_DualDWT(nn.Module):
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        IRDWTree:bool=False,
        Illu_scale:bool=False,
        CSA: bool=False,
        depth_scale:float = 1,
        dropout2d:float = 0
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        self.IRDWTree = IRDWTree
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)
        if "stem" in self.out_features:
            self.embedding1_rgb = nn.Sequential(
            BaseConv(9, stem_out_channels, ksize=3, act="lrelu", stride=1),
            BaseConv(stem_out_channels, stem_out_channels, ksize=3,  act="lrelu", stride=1),
            ) # 32
            self.embedding1_ir = nn.Sequential(
            BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
            BaseConv(stem_out_channels, stem_out_channels, ksize=3,  act="lrelu", stride=1),
            ) # 32
            if not CSA:
                self.CMRF1 = CMRFusion(stem_out_channels,stem_out_channels)
            else:
                self.CMRF1 = CSACMRFusion(stem_out_channels,stem_out_channels)

        in_channels = stem_out_channels * 2  # 64
        if "dark2" in self.out_features:
            self.embedding2_rgb = nn.Sequential(
                BaseConv(9, stem_out_channels, ksize=3, act="lrelu", stride=1),
                BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
                ) # 64
            self.embedding2_ir = nn.Sequential(
                BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
                BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
                ) # 64
            if not CSA:
                self.CMRF2 = CMRFusion(in_channels,in_channels)
            else:
                self.CMRF2 = CSACMRFusion(in_channels,in_channels)

        in_channels *= 2  # 128
        self.embedding3_rgb = nn.Sequential(
            BaseConv(9, stem_out_channels, ksize=3, act="lrelu", stride=1),
            BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1)
            ) # 128
        self.embedding3_ir = nn.Sequential(
            BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
            BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1)
            ) # 128
        if not CSA:
            self.CMRF3 = CMRFusion(in_channels,in_channels)
        else:
            self.CMRF3 = CSACMRFusion(in_channels,in_channels)

        in_channels *= 1  # 128->256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )

        in_channels *= 2  # 256
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )

        # IR wavelet tree
        self.Wavelet = DWT(wave='bior1.3')
        
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir, illu_aware = False):
        outputs = {}
        if self.IRDWTree:
            ir = ir2gray(ir) # b,c,h,w -> b,1,h,w
        # level 1
        ll,lh = self.Wavelet(ir,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        if "stem" in self.out_features:
            ir = self.embedding1_ir(ir)
        rgbll,rgblh = self.Wavelet(x,ignore_hh=True)
        rgblh = torch.flatten(rgblh,start_dim=1,end_dim=2)
        rgb = torch.cat([rgbll,rgblh],dim=1)
        if "stem" in self.out_features:
            rgb = self.embedding1_rgb(rgb)
            x = self.CMRF1(rgb,ir)
        else:
            x = torch.cat([rgb,ir],dim=1)
        outputs["stem"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
    
        # level 2
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        if "dark2" in self.out_features:
            ir = self.embedding2_ir(ir)
        rgbll,rgblh = self.Wavelet(rgbll,ignore_hh=True)
        rgblh = torch.flatten(rgblh,start_dim=1,end_dim=2)
        rgb = torch.cat([rgbll,rgblh],dim=1)
        if "dark2" in self.out_features:
            rgb = self.embedding2_rgb(rgb)
            x = self.CMRF2(rgb,ir)
        else:
            x = torch.cat([rgb,ir],dim=1)
        outputs["dark2"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 3
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding3_ir(ir)
        rgbll,rgblh = self.Wavelet(rgbll,ignore_hh=True)
        rgblh = torch.flatten(rgblh,start_dim=1,end_dim=2)
        rgb = torch.cat([rgbll,rgblh],dim=1)
        rgb = self.embedding3_rgb(rgb)
        x = self.CMRF3(rgb,ir)
        outputs["dark3"] = x
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 4
        x = self.dark4(x)
        outputs["dark4"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        # x = self.CMRF4(x,ir)

        # level 5
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}
    

class Darknet_RGBCNN(nn.Module):
    # implementation for dilated Cross Stage Concat
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        IRDWTree:bool=False,
        Illu_scale:bool=False,
        CSA: bool=False,
        depth_scale:float = 1,
        dropout2d:float = 0
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        self.IRDWTree = IRDWTree
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)

        self.stem = nn.Sequential( # backbone.backbone.stem
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu",not_pad=self.paddingCropped),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,not_pad=self.paddingCropped),
        )

        in_channels = stem_out_channels * 2  # 64
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential( # backbone.backbone.dark2
            *self.make_group_layer(in_channels, num_blocks[0], stride=2,not_pad=self.paddingCropped)
        )

        in_channels *= 2  # 128
        self.dark3 = nn.Sequential( # backbone.backbone.dark3
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )

        in_channels *= 2  # 256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        # self.embedding4 = nn.Sequential(
        #     BaseConv(3, 32, ksize=3, act="lrelu", stride=1),
        #     BaseConv(32, in_channels//2,ksize=3,  act="lrelu", stride=1),
        #     ) # 16
        # self.CMRF4 = CMRFusion(in_channels//2,in_channels*2)

        in_channels *= 2  # 512
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )
        
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir):
        outputs = {}
        
        # level 1
        x = self.stem(x)
        outputs["stem"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 2
        x = self.dark2(x)
        outputs["dark2"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 3
        x = self.dark3(x)
        outputs["dark3"] = x
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 4
        x = self.dark4(x)
        outputs["dark4"] = x
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 5
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}


class Darknet_IRDWT(nn.Module):
    
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        IRDWTree:bool=False,
        Illu_scale:bool=False,
        CSA: bool=False,
        depth_scale:float = 1,
        dropout2d:float = 0
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        self.IRDWTree = IRDWTree
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)
        # stem 
        in_channels = stem_out_channels * 2  # 64
        # dark 2
        in_channels *= 2  # 128
        # dark 3
        self.embedding3 = nn.Sequential(
            BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
            BaseConv(stem_out_channels, in_channels, ksize=3,  act="lrelu", stride=1),
            )
        in_channels *= 1  # 256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        # self.embedding4 = nn.Sequential(
        #     BaseConv(3, 32, ksize=3, act="lrelu", stride=1),
        #     BaseConv(32, in_channels//2,ksize=3,  act="lrelu", stride=1),
        #     ) # 16
        # self.CMRF4 = CMRFusion(in_channels//2,in_channels*2)

        in_channels *= 2  # 512
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )

        # IR wavelet tree
        self.Wavelet = DWT(wave='bior1.3')
        
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir, illu_aware = False):
        outputs = {}
        x=None
        if self.IRDWTree:
            ir = ir2gray(ir) # b,c,h,w -> b,1,h,w
        
        # level 1
        ll,_ = self.Wavelet(ir,ignore_hh=True)
        outputs["stem"] = ll
            
        # level 2
        ll,_ = self.Wavelet(ll,ignore_hh=True)
        outputs["dark2"] = ll
            
        # level 3
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        x = self.embedding3(ir)
        outputs["dark3"] = x
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 4
        x = self.dark4(x)
        outputs["dark4"] = x
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 5
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
class Darknet_Fusion_EarlyFusion(nn.Module):

    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        depth_scale:float = 1,
        dropout2d:float = 0
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)

        self.stem = nn.Sequential( # backbone.backbone.stem
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu",not_pad=self.paddingCropped),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,not_pad=self.paddingCropped),
        )

        in_channels = stem_out_channels * 2  # 64
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential( # backbone.backbone.dark2
            *self.make_group_layer(in_channels, num_blocks[0], stride=2,not_pad=self.paddingCropped)
        )

        in_channels *= 2  # 128
        self.dark3 = nn.Sequential( # backbone.backbone.dark3
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )

        in_channels *= 2  # 256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )

        in_channels *= 2  # 512
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )

        # IR wavelet tree
        self.Wavelet = DWT(wave='bior1.3')
        
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir, illu_aware = False):
        outputs = {}
        x = torch.cat([x,ir],dim=1)
        # level 1
        x = self.stem(x)
        outputs["stem"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 2
        x = self.dark2(x)
        outputs["dark2"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 3
        x = self.dark3(x)
        outputs["dark3"] = x
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 4
        x = self.dark4(x)
        outputs["dark4"] = x
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 5
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
class Darknet_Fusion_LateFusion(nn.Module):

    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        IRDWTree:bool=False,
        Illu_scale:bool=False,
        CSA: bool=False,
        depth_scale:float = 1,
        dropout2d:float = 0
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        self.IRDWTree = IRDWTree
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)

        self.stem = nn.Sequential( # backbone.backbone.stem
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu",not_pad=self.paddingCropped),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,not_pad=self.paddingCropped),
        )

        in_channels = stem_out_channels * 2  # 64
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential( # backbone.backbone.dark2
            *self.make_group_layer(in_channels, num_blocks[0], stride=2,not_pad=self.paddingCropped)
        )

        in_channels *= 2  # 128
        self.dark3 = nn.Sequential( # backbone.backbone.dark3
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        self.embedding3 = nn.Sequential(
            BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
            BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
            ) # 16
        self.LateFusion=nn.Sequential(
            BaseConv(in_channels*3, in_channels, ksize=1, act="lrelu", stride=1),
            BaseConv(in_channels, in_channels*2,ksize=3,  act="lrelu", stride=1),
            ) # 16

        in_channels *= 2  # 256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )

        in_channels *= 2  # 512
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )

        # IR wavelet tree
        self.Wavelet = DWT(wave='bior1.3')
        
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir, illu_aware = False):
        outputs = {}
        if self.IRDWTree:
            ir = ir2gray(ir) # b,c,h,w -> b,1,h,w
        
        # level 1
        ll,lh = self.Wavelet(ir,ignore_hh=True)
        x = self.stem(x)
        outputs["stem"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 2
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        x = self.dark2(x)
        outputs["dark2"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 3
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding3(ir)
        x = self.dark3(x)
        outputs["dark3"] = x
        x = torch.cat([x,ir],dim=1)
        x = self.LateFusion(x)
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 4
        x = self.dark4(x)
        outputs["dark4"] = x
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 5
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}

class Darknet_Fusion_MidCat(nn.Module):

    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        IRDWTree:bool=False,
        Illu_scale:bool=False,
        CSA: bool=False,
        depth_scale:float = 1,
        dropout2d:float = 0
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        self.IRDWTree = IRDWTree
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)

        self.stem = nn.Sequential( # backbone.backbone.stem
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu",not_pad=self.paddingCropped),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,not_pad=self.paddingCropped),
        )
        self.embedding1 = BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1)
        self.MidFusion1=nn.Sequential(
            BaseConv(stem_out_channels*3, stem_out_channels*2, ksize=1, act="lrelu", stride=1),
            ) # 16

 

        in_channels = stem_out_channels * 2  # 64
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential( # backbone.backbone.dark2
            *self.make_group_layer(in_channels, num_blocks[0], stride=2,not_pad=self.paddingCropped)
        )
        self.embedding2 = nn.Sequential(
            BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
            BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
            ) # 16
        self.MidFusion2=nn.Sequential(
            BaseConv(in_channels*3, in_channels*2, ksize=1, act="lrelu", stride=1),
            ) # 16

        in_channels *= 2  # 128
        self.dark3 = nn.Sequential( # backbone.backbone.dark3
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        self.embedding3 = nn.Sequential(
            BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
            BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
            ) # 16
        self.MidFusion3=nn.Sequential(
            BaseConv(in_channels*3, in_channels*2, ksize=1, act="lrelu", stride=1),
            ) # 16

        in_channels *= 2  # 256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        # self.embedding4 = nn.Sequential(
        #     BaseConv(3, 32, ksize=3, act="lrelu", stride=1),
        #     BaseConv(32, in_channels//2,ksize=3,  act="lrelu", stride=1),
        #     ) # 16
        # self.CMRF4 = CMRFusion(in_channels//2,in_channels*2)

        in_channels *= 2  # 512
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )

        # IR wavelet tree
        self.Wavelet = DWT(wave='bior1.3')
        
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir, illu_aware = False):
        outputs = {}
        if self.IRDWTree:
            ir = ir2gray(ir) # b,c,h,w -> b,1,h,w
        
        # level 1
        ll,lh = self.Wavelet(ir,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding1(ir)
        x = self.stem(x)
        outputs["stem"] = x
        x = torch.cat([x,ir],dim=1)
        x = self.MidFusion1(x)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 2
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding2(ir)
        x = self.dark2(x)
        outputs["dark2"] = x
        x = torch.cat([x,ir],dim=1)
        x = self.MidFusion2(x)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 3
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding3(ir)
        x = self.dark3(x)
        outputs["dark3"] = x
        x = torch.cat([x,ir],dim=1)
        x = self.MidFusion3(x)
        if self.is_dropout:
            x = self.dropout2d(x)

        # level 4
        # ll,lh = self.Wavelet(ll,ignore_hh=True)
        # lh = lh.squeeze(1)
        # ir = torch.cat([ll,lh],dim=1)
        # ir = self.embedding4(ir)
        x = self.dark4(x)
        outputs["dark4"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        # x = self.CMRF4(x,ir)

        # level 5
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
class Darknet_Fusion_DMAF(nn.Module):
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        IRDWTree:bool=False,
        depth_scale:float = 1,
        dropout2d:float = 0,
        CE:bool=True,
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        self.IRDWTree = IRDWTree
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)

        self.stem = nn.Sequential( # backbone.backbone.stem
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu",not_pad=self.paddingCropped),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,not_pad=self.paddingCropped),
        )
        if CE:
            self.embedding1 = BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1)
        else:
            self.embedding1 = nn.Conv2d(3, stem_out_channels, 1)   

        self.DMAF1 = DMAF(stem_out_channels,stem_out_channels*2)

        in_channels = stem_out_channels * 2  # 64
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential( # backbone.backbone.dark2
            *self.make_group_layer(in_channels, num_blocks[0], stride=2,not_pad=self.paddingCropped)
        )
        if CE:
            self.embedding2 = nn.Sequential(
                BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
                BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
                ) # 16
        else:
            self.embedding2 = nn.Conv2d(3, in_channels, 1)
        self.DMAF2 = DMAF(in_channels,in_channels*2)


        in_channels *= 2  # 128
        self.dark3 = nn.Sequential( # backbone.backbone.dark3
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        if CE:
            self.embedding3 = nn.Sequential(
                BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
                BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
                ) # 16
        else:
            self.embedding3 =nn.Conv2d(3, in_channels, 1)

        self.DMAF3 = DMAF(in_channels,in_channels*2)

        in_channels *= 2  # 256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )

        in_channels *= 2  # 512
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )

        # IR wavelet tree
        self.Wavelet = DWT(wave='bior1.3')
        
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir, illu_aware = False):
        outputs = {}
        if self.IRDWTree:
            ir = ir2gray(ir) # b,c,h,w -> b,1,h,w
        
        # level 1
        ll,lh = self.Wavelet(ir,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding1(ir)
        x = self.stem(x)
        outputs["stem"] = x
        x = self.DMAF1(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 2
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding2(ir)
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.DMAF2(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 3
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding3(ir)
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.DMAF3(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)

        x = self.dark4(x)
        outputs["dark4"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        # x = self.CMRF4(x,ir)

        # level 5
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
class Darknet_Fusion_InterMA(nn.Module):
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4], 33:[2, 4, 3, 2]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
        IRDWTree:bool=False,
        depth_scale:float = 1,
        dropout2d:float = 0,
        CE:bool=True,
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.paddingCropped = False
        self.out_features = out_features
        self.IRDWTree = IRDWTree
        num_blocks = self.depth2blocks[depth]
        stem_out_channels = int(stem_out_channels*depth_scale)

        self.stem = nn.Sequential( # backbone.backbone.stem
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu",not_pad=self.paddingCropped),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2,not_pad=self.paddingCropped),
        )
        if CE:
            self.embedding1 = BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1)
        else:
            self.embedding1 = nn.Conv2d(3, stem_out_channels, 1)   

        self.InterMA1 = InterMA(stem_out_channels,stem_out_channels*2)

        in_channels = stem_out_channels * 2  # 64
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential( # backbone.backbone.dark2
            *self.make_group_layer(in_channels, num_blocks[0], stride=2,not_pad=self.paddingCropped)
        )
        if CE:
            self.embedding2 = nn.Sequential(
                BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
                BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
                ) # 16
        else:
            self.embedding2 = nn.Conv2d(3, in_channels, 1)
        self.InterMA2 = InterMA(in_channels,in_channels*2)


        in_channels *= 2  # 128
        self.dark3 = nn.Sequential( # backbone.backbone.dark3
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        if CE:
            self.embedding3 = nn.Sequential(
                BaseConv(3, stem_out_channels, ksize=3, act="lrelu", stride=1),
                BaseConv(stem_out_channels, in_channels,ksize=3,  act="lrelu", stride=1),
                ) # 16
        else:
            self.embedding3 =nn.Conv2d(3, in_channels, 1)

        self.InterMA3 = InterMA(in_channels,in_channels*2)

        in_channels *= 2  # 256
        self.dark4 = nn.Sequential( # backbone.backbone.dark4
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )

        in_channels *= 2  # 512
        self.dark5 = nn.Sequential( # backbone.backbone.dark5
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
            )

        # IR wavelet tree
        self.Wavelet = DWT(wave='bior1.3')
        
        self.is_dropout = bool(dropout2d > 0)
        self.dropout2d = nn.Dropout2d(p=dropout2d, inplace=False) if self.is_dropout else nn.Identity()
    
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1, not_pad:bool = False):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu",not_pad=not_pad),
            *[(ResLayer(in_channels * 2,not_pad=not_pad)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x, ir, illu_aware = False):
        outputs = {}
        if self.IRDWTree:
            ir = ir2gray(ir) # b,c,h,w -> b,1,h,w
        
        # level 1
        ll,lh = self.Wavelet(ir,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding1(ir)
        x = self.stem(x)
        outputs["stem"] = x
        x = self.InterMA1(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 2
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding2(ir)
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.InterMA2(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)
            
        # level 3
        ll,lh = self.Wavelet(ll,ignore_hh=True)
        lh = lh.squeeze(1)
        ir = torch.cat([ll,lh],dim=1)
        ir = self.embedding3(ir)
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.InterMA3(x,ir)
        if self.is_dropout:
            x = self.dropout2d(x)

        x = self.dark4(x)
        outputs["dark4"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        # x = self.CMRF4(x,ir)

        # level 5
        x = self.dark5(x)
        outputs["dark5"] = x
        if self.is_dropout:
            x = self.dropout2d(x)
        
        return {k: v for k, v in outputs.items() if k in self.out_features}