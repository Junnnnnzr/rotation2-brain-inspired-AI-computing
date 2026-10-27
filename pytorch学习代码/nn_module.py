from torch import nn
import torch

class sjwl(nn.Module):
    def __init__(self):
        super(sjwl,self).__init__()

    def forward(self,input):
        output=input+1
        return output

sjwl1=sjwl()
x=torch.tensor(1.0)
output=sjwl1(x)
print(output)