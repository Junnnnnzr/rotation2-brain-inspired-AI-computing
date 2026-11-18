# 已有一个只能处理单个时间步 [N,*] 的模块，怎样让它处理完整时间序列 [T,N,*]？
# 1.函数风格：functional.multi_step_forward
# 2.模块风格：layer.MultiStepContainer

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional,neuron,layer
x_seq=[]
net_s=neuron.IFNode(step_mode='s')

y_seq=functional.multi_step_forward(x_seq,net_s)

net_s.reset()
net_m=layer.MultiStepContainer(net_s)
z_seq=net_m(x_seq)

