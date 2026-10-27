# Conv2d:用多个可学习的卷积核扫描输入图像，把原始像素转换为边缘，纹理和形状等特征图

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset", train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader =DataLoader(dataset,batch_size=64)

class sjwl(nn.Module):
    def __init__(self):
        super(sjwl,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        # in_channels 输入特征图的通道数（RGB图像有3个通道）
        # stride 卷积核每次移动的步长 默认为1 （2，1）可以表示高度方向移动2，宽度方向移动1
        # padding 在输入图像边缘添加像素的数量
    def forward(self,x):
        x=self.conv1(x)
        return x

sjwl1=sjwl()

writer=SummaryWriter("../logs")

step=0
for data in dataloader:
    imgs,targets=data
    output=sjwl1(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input",imgs,step)

    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step=step+1

