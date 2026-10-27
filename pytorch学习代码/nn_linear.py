# Linear: 全连接层，也叫线性层，把输入的一组特征进行加权组合和，转换成指定数量的输出特征

import torch
import torchvision
from torch import nn
from torch.nn import Linear


dataset=torchvision.datasets.CIFAR10("./dataset", train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=torch.utils.data.DataLoader(dataset,batch_size=64)

class sjwl(nn.Module):
    def __init__(self):
        super(sjwl,self).__init__()
        self.linear1=Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return output

sjwl1=sjwl()

for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    # output=torch.reshape(imgs,(1,1,1,-1))
    output=torch.flatten(imgs)
    print(output.shape)
    output=sjwl1(output)
    print(output.shape)
