# 在spikingjelly中怎么用pytorch张量表示“一组神经元有没有发放脉冲”

import torch

v=torch.rand([8]) # 生成长度为8的一维张量，理解成8个神经元当前的膜电位
v_th=0.5 # 发放阈值
spike=(v>=v_th).to(v) # .to(v)让判断结果与v有相同的数据类型，即float
print('spike=',spike)

# spike = tensor([0., 0., 0., 1., 1., 0., 1., 0.])

# 数据格式：
# 1. shape=[N, *]：一个时间步的数据
# 2. shape=[T, N, *]：多个时间步的数据
# T：时间步数量
# N：一次处理的样本数量
# *：每个样本剩余的数据维度，比如每个样本包含多少神经元的数据