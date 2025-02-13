#!/usr/bin/env python
# -*- encoding: utf-8 -*-
 

import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet_transform import DWT
from .CMRF import CMRFusion
import numpy as np
import pywt
import math
import torch.nn.init as init
from torch.nn.parameter import Parameter

#----------------------------------------IRDWTree----------------------------------------------#
def ir2gray(x:torch.Tensor):
        assert x.size(1) == 3, f"expect B,C,H,W input, instead of received {x.shape}"
        # Rec.ITU-R BT.601-7 Standard，BGR Sequence，for brightness calculation
        gray_score = torch.Tensor([0.1140, 0.5870 , 0.2989]).reshape((1,3,1,1)).to(x.device) 
        x = torch.sum(x*gray_score,dim=1,keepdim=True)
        return x
#----------------------------------------IRDWTree----------------------------------------------#

#----------------------------------------WaveConv----------------------------------------------#
def prep_wavelet_kernel(wave, inCh:int, device=None):
    """
    Prepares the wavelet filter kernel.  

    """
    if isinstance(wave, str):
        wave = pywt.Wavelet(wave)
    if isinstance(wave, pywt.Wavelet):
        h_col = wave.dec_hi
    t = torch.get_default_dtype()
    
    # h0_col = np.array(h0_col[::-1]).ravel() # low pass column filter bank
    h_col = np.array(h_col[::-1]).ravel() # high pass column filter bank   
    
    # h0_row = torch.tensor(h0_col, device=device, dtype=t).reshape((1, -1))
    # h0_col = torch.tensor(h0_col, device=device, dtype=t).reshape((-1, 1))
    h_row = torch.tensor(h_col, device=device, dtype=t).reshape((1, 1, 1, -1)).repeat(inCh,1,1,1)
    h_col = torch.tensor(h_col, device=device, dtype=t).reshape((1, 1, -1, 1)).repeat(inCh,1,1,1)
    
    # kernel_h0 = torch.matmul(h0_col,h0_row)
    # kernel_h1 = torch.matmul(h1_col,h1_row)
    return h_row, h_col

class wavelet_highpass_conv(nn.Module):
    """
    """
    def __init__(self, inCh, waves:list=['haar','bior1.3'],chReduction:int=4,learnable:bool=True):
        super().__init__()
        self.inCh = inCh
        self.chReduction = chReduction
        self.learnable = learnable
        self.compressor = nn.Conv2d(inCh,inCh//chReduction,kernel_size=1) # .flatten(start_dim=1,end_dim=2)
        self.waves = waves
        assert waves == ['haar','bior1.3'], f'not implement the forward function for {waves}'
        filters = [prep_wavelet_kernel(wave,inCh//chReduction) for wave in waves]
        # self.lowWave = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        for i in range(len(waves)):
            self.register_buffer(f'kernel_highWave_row{i}', filters[i][0])
            self.register_buffer(f'kernel_highWave_col{i}', filters[i][1])
            
        self.pad_row = [(1,0,0,0),(0,1,0,0)]
        self.pad_col = [(0,0,1,0),(0,0,0,1)]
        # if learnable:
            # self.affline_scale = Parameter(torch.Tensor(1, inCh//chReduction*2, 1, 1))
            # self.affline_bias = Parameter(torch.Tensor(1, inCh//chReduction*2, 1, 1))
        # self.fuse_conv = nn.Conv2d(inCh + inCh//chReduction*2,inCh,kernel_size=1,groups=inCh//chReduction)
        self.fuse_conv = nn.Conv2d(inCh + inCh//chReduction*2,inCh,kernel_size=3,padding=1)

    
    # def reset_parameters(self) -> None:
    #     init.kaiming_uniform_(self.affline_scale, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in = self.inCh//self.chReduction*2
    #         bound = 1 / math.sqrt(fan_in)
    #         init.uniform_(self.bias, -bound, bound)
    
    def forward(self,input):
        assert input.size(1)%(self.chReduction*4)==0, f'x@{x.size(1)} % 16 need to be 0'
        x = self.compressor(input) # ch = ch//4
        # DCT Low-pass extraction
        # x_low = self.lowWave(x)

        # symmetric padding refer to <<Convolution with even-sized kernels and symmetric padding>>
        xrow = list(torch.chunk(x,2,dim=1))
        xcol = list(torch.chunk(x,2,dim=1))
        for i in range(2):
            # padding 0 for high-pass wavelet filter will cause high noise at the edge
            # Thus the reflect or replicate padding should be used
            xrow[i] = F.pad(xrow[i],self.pad_row[i],mode='replicate')
            xcol[i] = F.pad(xcol[i],self.pad_col[i],mode='replicate')
        xrow = torch.cat(xrow,dim=1)
        xcol = torch.cat(xcol,dim=1)

        c = x.size(1)
        # DCT High-pass extraction
        # haar
        x_high_haar = F.conv2d(xrow, weight=self.kernel_highWave_row0,groups=c)
        x_high_haar = x_high_haar + F.conv2d(xcol, weight=self.kernel_highWave_col0,groups=c)
        # bior13
        xrow = F.pad(xrow,(2,2,0,0),mode='replicate')
        xcol = F.pad(xcol,(0,0,2,2),mode='replicate')
        x_high_bior13 = F.conv2d(xrow, weight=self.kernel_highWave_row1,groups=c)
        x_high_bior13 = x_high_bior13 + F.conv2d(xcol, weight=self.kernel_highWave_col1,groups=c)
        x = torch.stack([x_high_haar,x_high_bior13],dim=2).flatten(start_dim=1,end_dim=2) # 通道按个数混叠
        # x = self.fuse_conv(x) + input
        x = torch.cat([input,x],dim=1)
        x = self.fuse_conv(x)

        return x
#----------------------------------------WaveConv----------------------------------------------#

#----------------------------------------WavePool----------------------------------------------#
class wavelet_decmp(nn.Module):
    def __init__(self, waves:list=['haar','bior1.3'], chFuse_stack:bool=True):
        """
            Module for wavelet-based low&high-pass frequency decomposition.
            waves: list of str (wavelet name)
            channel_fuse: the frequency channel non-fused stack or fused stack
            
            Notes: Class DWT refer to DWTForward https://github.com/fbcotter/pytorch_wavelets
        """
        super().__init__()
        self.wave_list = nn.ModuleList([DWT(wave=w) for w in waves])
        self.fused_stack = chFuse_stack

    def forward(self,x):
        if self.fused_stack:
            # channel cat in the form: 1,2,3,4,1,2,3,4
            # for dwt in self.wave_list:
            #     out = dwt(x, only_low=True)
            x = torch.stack([dwt(x, only_low=True) for dwt in self.wave_list], dim=2)
            x = x.flatten(start_dim=1,end_dim=2)
        else:
            x= torch.cat([dwt(x, only_low=True) for dwt in self.wave_list], dim=1)
            
        return x

class BaseConv_WaveDown2(nn.Module):
    """A Conv2d partially applying wavelet kernel for high/low frequency decomposition and downsample
    -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, 
        ksize, stride, dilation=1, groups=1, bias=False, act="silu", 
        pad_mode:str = 'zeros',
        not_pad:bool = False,
        dual_groups:bool = False,
        waveDown_mode:bool = False,
    ):
        super().__init__()
        self.waveDown_mode = waveDown_mode
        self.dual = dual_groups
        self.waves = ['haar','bior1.3'] # the bior1.1 and db1 share the same lowpass kernel with haar
        if self.dual:
            groups=2

        self.not_pad = not_pad
        if self.not_pad:
            pad = 0
        else:
            pad = (ksize - 1) // 2
        
        if self.waveDown_mode:
            assert stride==2
            self.waveDown = wavelet_decmp(waves=self.waves, chFuse_stack=True)
            self.conv = nn.Conv2d(
                in_channels*len(self.waves),
                out_channels,
                kernel_size=ksize,
                bias=bias,
                stride=1,
                padding=pad,
            )
        else:            
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=ksize,
                stride=stride,
                dilation=dilation,
                padding=pad,
                padding_mode=pad_mode,
                groups=groups,
                bias=bias,
            )          
        if not self.dual:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bnR = nn.BatchNorm2d(out_channels//2)
            self.bnT = nn.BatchNorm2d(out_channels//2)
        self.act = get_activation(act, inplace=True)
        
    def forward(self, x):
        if self.waveDown_mode:
            x = self.conv(self.waveDown(x))
            x1, x2 =torch.chunk(x,2,dim=1)
            x = torch.cat([self.bnR(x1),self.bnT(x2)], dim=1)
            return self.act(x)
        else:
            if not self.dual:
                return self.act(self.bn(self.conv(x)))
            else:
                x = self.conv(x)
                x1, x2 =torch.chunk(x,2,dim=1)
                x = torch.cat([self.bnR(x1),self.bnT(x2)], dim=1)
                return self.act(x)

    def fuseforward(self, x):
        assert False, f'not supported!'
        return self.act(self.conv(x))
    
#----------------------------------------WavePool----------------------------------------------#

#----------------------------------------diffusion----------------------------------------------#  
class dual_diffusion(nn.Module):
    def __init__(self, in_ch:int, hw_size:int, learnable_score:bool=False, diff_on_DCT:bool=False, pad_mode:str = 'zeros'):
        """
            in_ch refer to the total channel of input feature maps.
            learnable_score: determine whether use the learnable fc
            diff_on_DCT: whether use channel-wise DCT to describe feature maps
        """
        super().__init__()
        in_ch = in_ch // 2
        self.learnable = learnable_score
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = sigmoid_2()
        self.size = hw_size
        self.DCT = diff_on_DCT
        if in_ch<=32:
            reduction_ratio = 4
        else:
            reduction_ratio = 16
        if diff_on_DCT:
            self.dctDiff_dual = DCTDiffAttn(channel=in_ch, dct_h=self.size, dct_w=self.size)

        if not diff_on_DCT and learnable_score:
            # self.fc = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
            self.fc = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // reduction_ratio, kernel_size=1, bias=False),
                nn.Conv2d(in_ch // reduction_ratio, in_ch, kernel_size=1, bias=False),
            )

        self.conv_rgb = nn.Sequential(
            BaseConv(in_ch, in_ch, ksize=3, stride=1), # default nn.Conv2d(in_ch, in_ch, kernel_size=3),
            #nn.Conv2d(in_ch // 2, in_ch, kernel_size=3, padding=1,padding_mode=pad_mode),
        )
        self.conv_ir = nn.Sequential(
            BaseConv(in_ch, in_ch, ksize=3, stride=1),
            #nn.Conv2d(in_ch // 2, in_ch, kernel_size=3, padding=1,padding_mode=pad_mode),
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        _,c,_,_ = x.shape
        rgb, ir = torch.chunk(x,chunks=2,dim=1)
        if self.DCT:
            score_rgb2ir, score_ir2rgb = self.dctDiff_dual(rgb, ir)
        elif not self.DCT and self.learnable:
            score_rgb2ir = self.act(
                    self.fc(self.act(self.avgpool(torch.sub(rgb, ir))))
                    )
            score_ir2rgb = self.act(
                    self.fc(self.act(self.avgpool(torch.sub(ir, rgb))))
                    )
        else:
            score_rgb2ir = self.act(self.avgpool(torch.sub(rgb, ir)))
            score_ir2rgb = self.act(self.avgpool(torch.sub(ir, rgb)))
            
        rgb = rgb + self.conv_rgb(ir * score_ir2rgb)
        ir = ir + self.conv_ir(rgb * score_rgb2ir)
        x = torch.cat([rgb,ir],dim=1)

        return x

class sigmoid_2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(input)-0.5)*2

class BiReFusion(nn.Module):
    # Bidirectional Reassemble Spatial Fusion for Multisepctral Features
    def __init__(self, in_ch:int, k_size:int=3):
        """
            相互采样并填补给对方
            in_ch refer to the total channel of input feature maps.
            learnable_score: determine whether use the learnable fc
            diff_on_DCT: whether use channel-wise DCT to describe feature maps
        """
        super().__init__()
        in_ch = in_ch // 2
        self.rgb_Biext = CMRFusion(in_ch,in_ch,k_size=k_size)
        self.ir_Biext = CMRFusion(in_ch,in_ch,k_size=k_size)


    def forward(self,x:torch.Tensor)->torch.Tensor:
        _,c,_,_ = x.shape
        rgb, ir = x[:,:c//2,:,:], x[:,c//2:,:,:]#torch.chunk(x,chunks=2,dim=1) # 导致无梯度
        rgb = self.rgb_Biext(rgb,ir)
        ir = self.ir_Biext(ir,rgb)   
        x = torch.cat([rgb,ir],dim=1)

        return x
#----------------------------------------diffusion----------------------------------------------#  


def padding_for_fpn(x:torch.Tensor, pad_n:int = 8):
    assert len(x.shape)==4, "input should have 4 dimensions for b,c,h,w."
    b,c,h,w = x.shape
    assert h==w, f'H {h}!= W {w} for input x'
    # 找最小的8的倍数
    if h%pad_n!=0:
        H = (h//pad_n)*pad_n+pad_n
        pad_len = H-h
        pad_less = pad_len // 2
        pad_kernel = (pad_less,pad_len-pad_less,pad_less,pad_len-pad_less)
        x = F.pad(x,pad_kernel,'constant',0) # same as the infer padding value for image: 114, 这个效果证明为不好
    return x

# TODO：考虑将padding改成crop来满足特征图改为8的倍数
def crop_for_fpn(x:torch.Tensor, pad_n:int = 8):
    assert len(x.shape)==4, "input should have 4 dimensions for b,c,h,w."
    b,c,h,w = x.shape
    assert h==w, f'H {h}!= W {w} for input x'
    # stem: f_out = f/2-4
    # dark2: f_out = f/4-7
    # dark3: f_out = f/8-4
    # dark4: f_out = f/16-2
    # dark5: f_out = f/32-1

    # 删除x的最右边特征图
    return x[:, :, 0:-1, 0:-1].contiguous()

def padding_7(x:torch.Tensor, pad_n:int = 8):
    assert len(x.shape)==4, "input should have 4 dimensions for b,c,h,w."
    _,_,h,w = x.shape
    assert h==w, f'H {h}!= W {w} for input x'
    # 进行padding_crop操作后，Stage2后的H' = H/4-7
    # 左padding 3，右padding 4
    pad_kernel = (3,4,3,4)
    x = F.pad(x,pad_kernel,'constant',0)
    return x

def center_crop(x):
    """
    center crop layer. crop [1:-2] to eliminate padding influence.
    Crop 1 element around the tensor for 3*3 conv kernel
    input x can be a Variable or Tensor
    """
    return x[:, :, 1:-1, 1:-1].contiguous()

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, 
        ksize, stride, dilation=1, groups=1, bias=False, act="silu", 
        pad_mode:str = 'zeros',
        not_pad:bool = False,
        dual_groups:bool = False,
    ):
        super().__init__()
        self.dual = dual_groups
        if self.dual:
            groups=2

        self.not_pad = not_pad
        if self.not_pad:
            pad = 0
        else:
            pad = (ksize - 1) // 2
            
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            dilation=dilation,
            padding=pad,
            padding_mode=pad_mode,
            groups=groups,
            bias=bias,
        )
        if not self.dual:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(out_channels//2)
            self.bn2 = nn.BatchNorm2d(out_channels//2)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        if not self.dual:
            return self.act(self.bn(self.conv(x)))
        else:
            x = self.conv(x)
            x1, x2 =torch.chunk(x,2,dim=1)
            x = torch.cat([self.bn1(x1),self.bn2(x2)], dim=1)
            return self.act(x)

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, 
                 in_channels: int, 
                 not_pad:bool=False,
                 pad_mode:str='zeros',
                 dual_groups:bool=False, 
                 ch_diff:bool=False, # channel differential fusion aware
                 chdiff_Lrscore:bool=True,
                 hw_size:int = 0,
                 diffDCT:bool=False,
                 DWTConv:bool=False,
                 CMRF:bool=False,
                 ):
        super().__init__()
        self.ch_diff = ch_diff
        self.not_pad = not_pad
        self.DWTConv = DWTConv
        self.dual_groups = dual_groups
        self.CMRF = CMRF

        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, pad_mode=pad_mode, act="lrelu", dual_groups=dual_groups, not_pad=not_pad
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, pad_mode=pad_mode, act="lrelu", dual_groups=dual_groups, not_pad=not_pad
        )
        self.bifusion=nn.Module()
        if ch_diff and not CMRF:
            self.diffusion = dual_diffusion(in_ch=in_channels, hw_size=hw_size, diff_on_DCT=diffDCT, learnable_score=chdiff_Lrscore, pad_mode=pad_mode)
        elif CMRF:
            self.bifusion = BiReFusion(in_ch=in_channels, k_size=3)
        
        if DWTConv:
            if dual_groups:
                reduction = 2
            self.dwt_conv = wavelet_highpass_conv(inCh=in_channels//reduction) 
            # only apply for IR channels

    def forward(self, x):
        if self.DWTConv and self.dual_groups:
            rgb,ir = torch.chunk(x,2,dim=1)
            c = x.size(1)
            # only apply for IR channels
            ir = self.dwt_conv(ir)
            x = torch.cat([rgb,ir],dim=1)

        out = self.layer2(self.layer1(x))
        x = x+out        
        
        if self.ch_diff and not self.CMRF:
            x = self.diffusion(x)
        elif self.CMRF:
            x = self.bifusion(x)

        if self.not_pad:
            x = center_crop(x)
        return x
    

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu",pad_mode:str='zeros', dual:bool=False
    ):
        super().__init__()
        self.dual = dual
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation,pad_mode=pad_mode, dual_groups=dual)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation,pad_mode=pad_mode, dual_groups=dual)

    def forward(self, x):
        x = self.conv1(x)
        if not self.dual:
            x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        else:
            x1,x2 = torch.chunk(x,2,dim=1)
            x1 = torch.cat([x1] + [m(x1) for m in self.m], dim=1)
            x2 = torch.cat([x2] + [m(x2) for m in self.m], dim=1)
            x = torch.cat([x1, x2],dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
    

# class BaseConv(nn.Module):
#     """A Conv2d -> Batchnorm -> silu/leaky relu block"""

#     def __init__(
#         self, in_channels, out_channels, 
#         ksize, stride, dilation=1, groups=1, bias=False, act="silu", 
#         not_pad:bool = False,
#         dual_groups:bool = False,
#     ):
#         self.dual = dual_groups
#         if self.dual:
#             groups=2
#         super().__init__()
#         self.not_pad = not_pad
#         if self.not_pad:
#             pad = 0
#         else:
#             pad = (ksize - 1) // 2
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=ksize,
#             stride=stride,
#             dilation=dilation,
#             padding=pad,
#             groups=groups,
#             bias=bias,
#         )
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = get_activation(act, inplace=True)

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def fuseforward(self, x):
#         return self.act(self.conv(x))