import torch
import torchvision.ops as ops
from torch import nn
from torch.nn import functional as F

class sigmoid_2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(input)-0.5)*2
    
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
    
class DMAF(nn.Module):
    def __init__(self, channels_in:int, channels_o:int):
        """
            in_ch refer to the total channel of input feature maps.
            learnable_score: determine whether use the learnable fc
            diff_on_DCT: whether use channel-wise DCT to describe feature maps
        """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Tanh()

        self.embed = ConvBNReLU(channels_in, channels_o, kernel_size=1,stride=1,padding=0,dilation=1)
        self.fuse_out = ConvBNReLU(channels_o*2, channels_o, kernel_size=3,stride=1,padding=1,dilation=1)

    def forward(self, X_O:torch.Tensor, X_in:torch.Tensor)->torch.Tensor:
        b, c, h, w = X_in.size()
        _, C, H, W = X_O.size()
        # embed to same dimension
        X_in_E = self.embed(X_in)
        score_rgb2ir = self.act(self.avgpool(torch.sub(X_O, X_in_E)))
        score_ir2rgb = self.act(self.avgpool(torch.sub(X_in_E, X_O)))
            
        rgb = X_O + X_in_E * score_ir2rgb
        ir = X_in_E + X_O * score_rgb2ir
        x = torch.cat([rgb,ir],dim=1)
        x = self.fuse_out(x)

        return x

class InterMA(nn.Module):
    def __init__(self, channels_in:int, channels_o:int):
        """
            in_ch refer to the total channel of input feature maps.
            learnable_score: determine whether use the learnable fc
            diff_on_DCT: whether use channel-wise DCT to describe feature maps
        """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Tanh()

        self.embed = ConvBNReLU(channels_in, channels_o, kernel_size=1,stride=1,padding=0,dilation=1)
        self.FC = nn.Sequential(
            nn.Linear(channels_o,channels_o//4),
            nn.Linear(channels_o//4,channels_o),
        )
        self.fuse_out = ConvBNReLU(channels_o*2, channels_o, kernel_size=3,stride=1,padding=1,dilation=1)

    def forward(self, X_O:torch.Tensor, X_in:torch.Tensor)->torch.Tensor:
        b, c, h, w = X_in.size()
        _, C, H, W = X_O.size()
        # embed to same dimension
        X_in_E = self.embed(X_in)
        score_rgb2ir = self.act(self.FC(torch.squeeze(self.avgpool(torch.sub(X_O, X_in_E)))))
        score_ir2rgb = self.act(self.FC(torch.squeeze(self.avgpool(torch.sub(X_in_E, X_O)))))
        if len(score_rgb2ir.shape)==1:
            score_rgb2ir = score_rgb2ir.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            score_ir2rgb = score_ir2rgb.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            score_rgb2ir = score_rgb2ir.unsqueeze(2).unsqueeze(3)
            score_ir2rgb = score_ir2rgb.unsqueeze(2).unsqueeze(3)           
        rgb = X_O + X_in_E * score_ir2rgb
        ir = X_in_E + X_O * score_rgb2ir
        x = torch.cat([rgb,ir],dim=1)
        x = self.fuse_out(x)

        return x

class BAAGate(nn.Module):
    def __init__(self, channels_in:int, channels_o:int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = sigmoid_2()

    def forward(self, X_O:torch.Tensor, X_in:torch.Tensor)->torch.Tensor:
        b, c, h, w = X_in.size()
        _, C, H, W = X_O.size()

        return None