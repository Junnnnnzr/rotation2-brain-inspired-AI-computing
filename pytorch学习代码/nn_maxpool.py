# MaxPool2d: 在每个局部区域中只保留最大值，从而缩小特征图尺寸，并保留比较明显的特征，减少计算量

from torch import nn
import torch
from torch.nn import MaxPool2d
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor())
dataloader=torch.utils.data.DataLoader(dataset,batch_size=64)

# input=torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [1,2,1,0,0],
#                     [5,2,3,1,1],
#                     [2,1,0,1,1]],dtype=torch.float32)
#
# input=torch.reshape(input,(-1,1,5,5)) 将input转换成能接收的四维输入[N,C,H,W]
# print(input.shape)

class sjwl(nn.Module):
    def __init__(self):
        super(sjwl,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output=self.maxpool1(input)
        return output

sjwl1=sjwl()
# output=sjwl1(input)
# print(output)

writer=SummaryWriter("logs_maxpool")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=sjwl1(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()