# 整个SNN网络的传播顺序
# 1.逐步传播：先把时间步t=0走完整个网络，再处理t=1
# 2.逐层传播：先让所有时间步通过第一层，再让所有时间步通过第二层
# 在使用梯度替代法训练时，推荐使用逐层传播；在内存受限时使用逐步传播
import torch

# 逐步传播
for t in range(T): # 遍历所有时间步
    x=x_seq[t] # 取出第t个时间步的数据
    y=net(x) # 让当前时间步的数据通过完整网络
    y_seq_step_by_step.append(y.unsqueeze(0))

y_seq_step_by_step=torch.cat(y_seq_step_by_step,dim=0)
# T个 [1, N, *]变成1个 [T, N, *]

# 逐层传播
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer
T=4
N=2
C=8
x_seq=torch.rand([T,N,C])*64 # 时间步，批次大小，每个样本包含特征

net=nn.Sequential(
    layer.Linear(C,4),
    neuron.IFNode(),
    layer.Linear(4,2),
    neuron.IFNode()
)
functional.set_step_mode(net,step_mode='m')
with torch.no_grad():
    y_seq_layer_by_layer=x_seq
    for i in range(net.__len__()): # 遍历网络中的每一层
        y_seq_layer_by_layer=net[i](y_seq_layer_by_layer)

# 实际上最后一部分可以直接写成：y_seq_layer_by_layer = net(x_seq)