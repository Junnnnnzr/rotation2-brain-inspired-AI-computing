# 同一段包含多个时间步的数据，如何分别使用“单步模式”和“多步模式”送入 SpikingJelly 神经元
# 这不是两种不同的神经元，而是两种处理时间序列的方式。
# 1.单步模式 step_mode='s'：每次只处理一个时间步；
# 2.多步模式 step_mode='m'：一次处理完整的 T 个时间步

import torch
from spikingjelly.activation_based import neuron,functional

# 设置数据各个维度的大小
T = 4
N = 1
C = 3
H = 8
W = 8

# 创建包含多个时间步的输入数据
x_seq = torch.rand([T, N, C, H, W])

# 方法一：单步模式:如果我们想给单步模式的模块输入 shape = [T, N, *] 的序列数据，通常需要手动将数据拆成 T 个 shape = [N, *] 的数据并逐步输入进去
net_s=neuron.IFNode(step_mode='s') # 创建一个IF神经元，并设置为单步模式
y_seq=[] # 创建输出列表
for t in range(T):
    x = x_seq[t]  # x.shape = [N, C, H, W]
    y = net_s(x)  # y.shape = [N, C, H, W]
    y_seq.append(y.unsqueeze(0)) # unsqueeze(0)在y最前面增加一个维度

y_seq=torch.cat(y_seq) # cat用于将他们拼接起来

# 方法二：单步模式，使用 multi_step_forward 自动循环
y_seq1=functional.multi_step_forward(x_seq,net_s)

# 方法三：多步模式，直接输入完整时间序列
net_m=neuron.IFNode(step_mode='m')
y_seq2= net_m(x_seq)