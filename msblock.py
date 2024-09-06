import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0,groups=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False,
                groups=groups)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class MSBlockLayer(nn.Module):

    def __init__(self,in_channel,out_channel,kernel_size):
        super().__init__()
        self.in_conv = ConvBNReLU(in_channel,out_channel,1)
        self.mid_conv =ConvBNReLU(out_channel,out_channel,kernel_size,padding=kernel_size//2,groups=out_channel)
        self.out_conv =ConvBNReLU(out_channel,in_channel,1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.mid_conv(x)
        x = self.out_conv(x)
        return x




class MSBlock(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes, stride=1,in_expand_ratio=3., mid_expand_ratio=2., layers_num=3,
                 in_down_ratio=2. )-> None:
        super().__init__()
        # 根据扩展比例计算中间通道数
        in_channel = int(inc * in_expand_ratio // in_down_ratio)
        self.mid_channel = in_channel // len(kernel_sizes)
        groups = int(self.mid_channel * mid_expand_ratio)
        # 输入卷积层
        self.in_conv = ConvBNReLU(inc, in_channel,stride=stride)

        self.mid_convs = []
        # 根据给定的核大小创建多个MSBlockLayer
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel, groups, kernel_size=kernel_size) for _ in range(int(layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        # 输出卷积层
        self.out_conv = ConvBNReLU(in_channel, ouc, kernel_size=1)

        self.attention = None

    def forward(self, x):
        out = self.in_conv(x)
        channels = []
        # 分别处理每个通道范围内的特征，并合并
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i * self.mid_channel:(i + 1) * self.mid_channel, ...]
            if i >= 1:
                channel = channel + channels[i - 1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)
        return out# MSBlock模块，包含多个MSBlockLayer，用于处理不同尺度的特征


if __name__ == "__main__":
    in_ten = torch.randn(2, 80, 224, 224)
    conv2 = MSBlock(80, 80, kernel_sizes=[1, 3, 3], stride=2)   #have dowmsample
    # conv3 = MSBlock(80, 160, kernel_sizes=[1, 5, 5], stride=2)
    # conv4 = MSBlock(160, 256, kernel_sizes=[1, 7, 7], stride=2)

    out=conv2(in_ten)
    print(out.shape)