import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

class BasicConv(nn.Module):
    """
    Basic Convolutional block with optional BatchNorm and ReLU.
    """
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    """
    ZPool operation that combines max and average pooling.
    """
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    """
    Attention Gate block that modulates the input with a learned attention map.
    """
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class SFHead(nn.Module):
    """
    Spatial-Temporal Extraction and Adaptive Cross-dimensional Feature Aggregation block.
    """
    def __init__(self, channel=512, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.gn = nn.GroupNorm(channel // (4 * G), channel // (4 * G))  # GroupNorm

        # Learnable parameters
        self.cweight = Parameter(torch.zeros(1, channel // (4 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (4 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (4 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (4 * G), 1, 1))

        self.w = Parameter(torch.ones(4))

        self.sigmoid = nn.Sigmoid()
        self.cv = AttentionGate()
        self.tc = AttentionGate()
        self.tv = AttentionGate()
        self.bn = nn.BatchNorm2d(channel // (4 * G))
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """
        Initialize weights for Conv2d, BatchNorm2d, and Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        """
        Shuffle channels in groups for better feature representation.
        """
        b, c, t, v = x.shape
        x = x.reshape(b, groups, -1, t, v)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, t, v)
        return x

    def forward(self, x):
        """
        Forward pass through the Spatial-Temporal Extraction and Adaptive Cross-dimensional Feature Aggregation layers.
        
        b: number of training samples in the current batch (e.g. N=32)
        c: number of dimensions for skeleton representions (e.g. the initial C=3 for 3D keypoint)
        t: number of frames (e.g. T=300)
        v: number of keypoints (e.g. V=25 for NTU RGB+D 3D skeleton)
        
        """

        b, c, t, v = x.size()  
        
        x = x.view(b * self.G, -1, t, v)
        x_1, x_2, x_3, x_4 = x.chunk(4, dim=1)

        f_c = x_2
        f_o = x_4

        # Spatial-Temporal Extraction (SSTE)
        f_t = self.avg_pool(x_1)
        f_t = self.cweight * f_t + self.cbias
        f_t = x_1 * self.sigmoid(f_t)

        f_s = self.gn(x_3)
        f_s = self.sweight * f_s + self.sbias
        f_s = x_3 * self.sigmoid(f_s)

        # Adaptive Cross-dimensional Feature Aggregation (AC-FA)
        x_perm1 = f_t.permute(0, 2, 1, 3).contiguous()  # (N, C, T, V) -> (N, T, C, V) 
        x_out1 = self.cv(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = f_s.permute(0, 3, 2, 1).contiguous() # (N, C, T, V) -> (N, V, T, C) 
        x_out2 = self.tc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        x_out31 = self.tv(f_c) # (N, C, T, V)

        softmax_weights = torch.nn.functional.softmax(self.w, dim=0)

        f_r = (softmax_weights[0] * x_out11 + 
       softmax_weights[1] * x_out21 + 
       softmax_weights[2] * x_out31 + 
       softmax_weights[3] * f_c)


        f_a = torch.cat([f_t, f_s, f_r, f_o], dim=1)
        f_a = f_a.contiguous().view(b, -1, t, v)
        f_a = self.channel_shuffle(f_a, 2)

        return f_a


if __name__ == '__main__':
    '''
    Choose `channel` and `G` based on the following rules:
    - `channel` must be divisible by 4 * G (for GN) and by G (for channel shuffle).
    - Example:
        - For `G = 8`, valid `channel` values: 32, 64, 128, 256, 512, 1024, etc.
        - For `G = 4`, valid `channel` values: 16, 32, 64, 128, 256, etc.
        - For `G = 16`, valid `channel` values: 64, 128, 256, 512, 1024, etc.

    '''
    input = torch.randn(96, 64, 300, 25) 
    
    sf_head = SFHead(channel=64, G=8) 
    output = sf_head(input) 
    
    print(output.shape)  