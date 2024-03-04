import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, num_classes, input_size):
        super(DenseNet, self).__init__()

        # 加载预训练的DenseNet模型
        densenet = models.densenet201()
        self.densenet = models.densenet201()
        
        # 获取DenseNet的特征提取部分
        feature_extractor = nn.Sequential(*list(densenet.children())[:-1])
        
        # 获取特征提取部分的输出通道数
        in_channels = densenet.classifier.in_features
        
        # 定义自定义的分割头部，输出通道数为1
        segmentation_head = SegmentationHead(in_channels, num_classes)

        self.feature_extractor = feature_extractor
        self.segmentation_head = segmentation_head
        
        # 定义上采样层，用于将特征图的尺寸增加到输入尺寸
        self.upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(1, 1, kernel_size=(1000, 400))

    def forward(self, x):
        output = self.densenet(x)

        return output


# 定义自定义的分割头部
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


if __name__ == '__main__':

    input_size = (1000, 400)
    # 创建模型实例并训练
    model = DenseNet(1, input_size)

    # 创建一个随机输入
    input_tensor = torch.randn(1, 3, input_size[0], input_size[1])

    # 将输入传递给模型进行前向传播
    densenet = models.densenet201()
    output_tensor = model(input_tensor)

    # 打印输出尺寸
    print(output_tensor.size())
