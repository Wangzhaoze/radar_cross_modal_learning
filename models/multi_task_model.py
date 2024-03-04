""" Parts of the U-Net model """
"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        

class multi_task_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(multi_task_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up_seg1 = Up(1024, 256, bilinear)
        self.up_seg2 = Up(512, 128, bilinear)
        self.up_seg3 = Up(256, 64, bilinear)
        self.up_seg4 = Up(128, 64, bilinear)
        self.outc_seg = OutConv(64, 1)

        self.up_traj1 = Up(1024, 256, bilinear)
        self.up_traj2 = Up(512, 128, bilinear)
        self.up_traj3 = Up(256, 64, bilinear)
        self.up_traj4 = Up(128, 64, bilinear)
        self.outc_traj = OutConv(64, 1)
        self.sig = nn.Sigmoid()



    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x_seg = self.up_seg1(x5, x4)
        x_seg = self.up_seg2(x_seg, x3)
        x_seg = self.up_seg3(x_seg, x2)
        x_seg = self.up_seg4(x_seg, x1)
        logits_seg = self.outc_seg(x_seg)


        x_traj = self.up_seg1(x5, x4)
        x_traj = self.up_seg2(x_traj, x3)
        x_traj = self.up_seg3(x_traj, x2)
        x_traj = self.up_seg4(x_traj, x1)
        x_traj = self.outc_traj(x_traj)
        logits_traj = self.sig(x_traj)


        logits = torch.cat([logits_seg, logits_traj], dim=1)

        return logits


if __name__ == '__main__':
    net = multi_task_UNet(n_channels=3, n_classes=2)
    tensor = torch.zeros((1, 3, 640, 480))
    print(net(tensor).shape)