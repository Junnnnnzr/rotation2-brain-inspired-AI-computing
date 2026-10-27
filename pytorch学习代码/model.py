# 搭建神经网络
import torch
from torch import nn


class sjwl(nn.Module):
    def __init__(self):
        super(sjwl,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )


    def forward(self,x):
        x=self.model(x)
        return x

if __name__=="__main__":
    sjwl1=sjwl()
    input=torch.ones((64,3,32,32))
    output=sjwl1(input)
    print(output.shape)