# 加载现有的网络模型以及修改模型

import torchvision
from torchvision.models import VGG16_Weights
from torch import nn

vgg16_false=torchvision.models.vgg16(weights=None) # 只创建VGG16结构，权重随机初始化，需要自己训练
vgg16_true=torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT) # 加载imagenet预训练权重

print(vgg16_true)

train_data=torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)

# vgg16输出有1000类，CIFAR10输出有10类，想让二者匹配需要进行模型修改
# 1.在vgg16的classifier这一层最后加入一个线性层，输入为1000，输出为10
vgg16_true.classfier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)

# 2.直接修改vgg16的classifier的最后一层，输入为4096，输出为10
vgg16_false.classfier[6]=nn.Linear(4096,10)