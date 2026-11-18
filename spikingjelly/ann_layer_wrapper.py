# 普通的torch.nn.Conv2d等模块只能接受单时间步，无法接受SNN的多时间步数据，因此需要包装器逐时间步调用普通ANN层

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer

with torch.no_grad():
    T = 4
    N = 1
    C = 3
    H = 8
    W = 8
    x_seq = torch.rand([T, N, C, H, W])

    conv = nn.Conv2d(C, 8, kernel_size=3, padding=1, bias=False)
    bn = nn.BatchNorm2d(8)
    y_seq = functional.multi_step_forward(x_seq, (conv, bn))
    # 将conv和bn作为一个模块序列，逐时间步执行
    net = layer.MultiStepContainer(conv, bn)
    # 将卷积层和bn一起包装成一个多步模块
    z_seq = net(x_seq)