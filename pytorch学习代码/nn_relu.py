# ReLu/Sigmoid: 激活函数，给神经网络加入非线性，使得学习更加复杂的特征

import torch
import torchvision
from torch import nn
from torch.nn import Sigmoid
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.5],
                    [-1,3]])

input=torch.reshape(input,(-1,1,2,2))
print(input.shape)

dataset=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor())
dataloader=torch.utils.data.DataLoader(dataset,batch_size=64)

class sjwl(nn.Module):
    def __init__(self):
        super(sjwl,self).__init__()
        self.relu1=nn.ReLU()
        self.sigmoid1=Sigmoid()

    def forward(self, input):
        output=self.sigmoid1(input)
        return output

sjwl1=sjwl()
# output=sjwl1(input)
# print(output)

writer=SummaryWriter("logs_relu")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=sjwl1(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()