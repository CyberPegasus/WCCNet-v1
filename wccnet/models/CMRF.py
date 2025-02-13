
import torch
import torchvision.ops as ops
from torch import nn
from torch.nn import functional as F
    
class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class CMRFusion(nn.Module):
    def __init__(self, channels_in:int, channels_o:int, c_mid:int=64, k_size:int=3, k_enc:int=3, shortcut:bool=True,illu_scale:torch.Tensor=None):
        from .network_blocks import BaseConv
        """ 
        The fusion of same size
        Args:
            channels_in: The channel number of input features.
            channels_o: The channel number of origin features.
            c_mid: The channel number after compression.
            k_size: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.

        Returns:
            X: The upsampled feature map.
        """
        super(CMRFusion, self).__init__()
        self.kernel = k_size
        self.illu_scale = illu_scale
        c_mid = channels_o // 4 if channels_o > 64 else channels_o // 2
        self.shortcut = shortcut

        self.enc =  nn.Sequential(
                ConvBNReLU(channels_o, c_mid, kernel_size=k_enc, 
                            stride=1, padding=k_enc//2, dilation=1, 
                            use_relu=False),
                ConvBNReLU(c_mid, k_size**2, kernel_size=k_enc, 
                            stride=1, padding=k_enc//2, dilation=1, 
                            use_relu=False)   
        )
        self.unfold = nn.Unfold(kernel_size=k_size, dilation=1, stride=1,
                        padding=k_size//2)

        if self.shortcut:
            self.fuse_out = BaseConv(channels_o+channels_in*2, channels_o, 3, stride=1, act="lrelu")
        else:
            self.fuse_out = BaseConv(channels_o+channels_in, channels_o, 3, stride=1, act="lrelu")
            
    def forward(self, X_O:torch.Tensor, X_in:torch.Tensor, debug_vis:bool=False):
        # H = s * h, where s denotes the upsampling scale
        b, c, h, w = X_in.size()
        _, C, H, W = X_O.size()
        # assert the size match
        assert h==H and w==W, f'upscale not match {h}!={H}'
        
        # generate kernel map by X_hr with the same size as X_hr
        Kernel_down = self.enc(X_O)                       # b * k_up^2 * h * w
        Kernel_down = F.softmax(Kernel_down, dim=1)               # b * k_up^2 * h * w
        # Debug for visualization
        if debug_vis:
            vis_attention = Kernel_down
        
        X_down = self.unfold(X_in)         # b * (k_up^2*c) * (h * w)
        X_down = X_down.view(b, -1, h, w).contiguous()       # b * (k_up^2*c) * h * w
        X_down = X_down.view(b, c, -1, h, w).contiguous()      # b * c * k_up^2 * h * w
        # Kernel_down 作为加权的权重
        X_down = torch.einsum('bkhw,bckhw->bchw', [Kernel_down, X_down])   # b * c * H * W

        if self.shortcut:
            X_out = torch.cat([X_O, X_down, X_in],dim=1)
        else:
            X_out = torch.cat([X_O, X_down],dim=1)
            
        X_out = self.fuse_out(X_out)

        if not debug_vis:
            return X_out
        else:
            return X_out, vis_attention

class CSACMRFusion(nn.Module): # Bidirectional Reassembly CSA Spatial Fusion
    def __init__(self, channels_in:int, channels_o:int, c_mid:int=64, k_size:int=3, k_enc:int=3, shortcut:bool=True, mask:bool=False, debug_vis:bool=False):
        from .network_blocks import BaseConv
        """ 
        The fusion of same size
        Args:
            channels_in: The channel number of input features.
            channels_o: The channel number of origin features.
            c_mid: The channel number after compression.
            k_size: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.

        Returns:
            X: The upsampled feature map.
        """
        super().__init__()
        self.k_size=k_size
        self.is_mask = mask
        # 1. Offset Prediction for unaligned probelm
        self.cat_fuse_conv = BaseConv(channels_o+channels_in, channels_o, 3, stride=1, act="lrelu")
        self.offset_out_conv = nn.Conv2d(channels_o,2*(k_size**2),k_size,padding=k_size//2)
        nn.init.constant_(self.offset_out_conv.weight, 0.)
        nn.init.constant_(self.offset_out_conv.bias, 0.)
        if mask:
            self.mask_conv = nn.Conv2d(channels_o,k_size**2,k_size,padding=k_size//2)
            nn.init.constant_(self.mask_conv.weight, 0.)
            nn.init.constant_(self.mask_conv.bias, 0.)       
        self.aligned_conv = nn.Conv2d(channels_in,channels_in,k_size,padding=k_size//2)
        
        # 2. Reassembly Spatial Fusion
        c_mid = channels_o // 4 if channels_o > 64 else channels_o // 2
        self.shortcut = shortcut

        self.enc =  nn.Sequential(
                ConvBNReLU(channels_o, c_mid, kernel_size=k_enc, 
                            stride=1, padding=k_enc//2, dilation=1, 
                            use_relu=False),
                ConvBNReLU(c_mid, k_size**2, kernel_size=k_enc, 
                            stride=1, padding=k_enc//2, dilation=1, 
                            use_relu=False)   
        )
        self.unfold = nn.Unfold(kernel_size=k_size, dilation=1, stride=1,
                        padding=k_size//2)

        if self.shortcut:
            self.fuse_out = BaseConv(channels_o+channels_in*2, channels_o, 3, stride=1, act="lrelu")
        else:
            self.fuse_out = BaseConv(channels_o+channels_in, channels_o, 3, stride=1, act="lrelu")
            
    def forward(self, X_O:torch.Tensor, X_in:torch.Tensor, debug_vis:bool=False):
        b, c, h, w = X_in.size()
        _, C, H, W = X_O.size()
        # assert the size match
        assert h==H and w==W, f'upscale not match {h}!={H}'
        # 1. Offset Prediction for unaligned probelm
        X_off = torch.cat([X_O,X_in],dim=1)
        X_off = self.cat_fuse_conv(X_off)
        max_offset = max(h,w)/4
        offset = self.offset_out_conv(X_off).clamp(-max_offset, max_offset)
        if self.is_mask:
            mask = 2. * torch.sigmoid(self.mask_conv(X_off))

        if self.is_mask:
            X_align = ops.deform_conv2d(input=X_in,
                                        offset=offset,
                                        weight=self.aligned_conv.weight,
                                        bias=self.aligned_conv.bias,
                                        padding=self.k_size//2,
                                        mask=mask
                                        )
        else:
            X_align = ops.deform_conv2d(input=X_in,
                                        offset=offset,
                                        weight=self.aligned_conv.weight,
                                        bias=self.aligned_conv.bias,
                                        padding=self.k_size//2,
                                        )         
        
        # generate kernel map by X_hr with the same size as X_hr
        Kernel_down = self.enc(X_off)                       # b * k_up^2 * h * w
        Kernel_down = F.softmax(Kernel_down, dim=1)               # b * k_up^2 * h * w
        # Kernel_down 作为加权的权重
        if debug_vis:
            vis_attention = Kernel_down

        X_down = self.unfold(X_align)         # b * (k_up^2*c) * (h * w)
        X_down = X_down.view(b, -1, h, w).contiguous()       # b * (k_up^2*c) * h * w
        X_down = X_down.view(b, c, -1, h, w).contiguous()      # b * c * k_up^2 * h * w
        X_down = torch.einsum('bkhw,bckhw->bchw', [Kernel_down, X_down])   # b * c * H * W

        if self.shortcut:
            X_out = torch.cat([X_off, X_align, X_down],dim=1)
        else:
            X_out = torch.cat([X_off, X_align],dim=1)
        X_out = self.fuse_out(X_out)

        if not debug_vis:
            return X_out
        else:
            return X_out, vis_attention
    