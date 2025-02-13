# Illumination Aware Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet_transform import DWT

# Our implementation for Illumination Aware Module
# Deprecated in this repos
class illumination_aware(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wavelet = DWT(wave='bior1.3')
        self.avg_pool = nn.AdaptiveAvgPool2d((8,8))
        self.max_pool = nn.AdaptiveMaxPool2d((8,8))
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=3,kernel_size=3,stride=1),
            nn.LeakyReLU(inplace=True),
            # nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3,stride=1),
            nn.LeakyReLU(inplace=True),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=16,out_features=4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=4,out_features=2), # [night, day]
            nn.Sigmoid()
        )
        self.loss = nn.CrossEntropyLoss() #nn.BCELoss() #
        
        
    def forward(self, x:torch.Tensor, down_scale:int = 4):
        """
            Input RGB Image for Illumination Aware
        """
        for _ in range(down_scale):
            x,_ = self.Wavelet(x,ignore_hh=True) # b,3,h/16,w/16
        x = torch.cat([self.avg_pool(x),self.max_pool(x)],dim=1) # b,6,8,8
        x = self.conv_layer(x) # b,1,4,4
        x = x.flatten(start_dim=1) # b,16
        x = self.fc_layer(x) # b,2
        return x

    def get_loss(self, outputs, labels):
        return self.loss(outputs, labels)
