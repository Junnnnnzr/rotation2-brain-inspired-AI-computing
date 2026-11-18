import torch
from spikingjelly.activation_based import neuron
from spikingjelly import  visualizing
from matplotlib import pyplot as plt

if_layer=neuron.IFNode()
# if神经元的一些构造参数：
# 1.v_threshold  神经元的阈值电压
# 2.v_reset 神经元的重置电压。若为None，电压减去阈值；反之重置为此
# 3.surrogate_function 反向传播时用来计算脉冲函数梯度的替代函数

print(if_layer.v)
# 膜电位v，初始为0因为没有任何输入

x=torch.rand(size=[2,3])
if_layer(x)
print(f'x.shape={x.shape}, if_layer.v.shape={if_layer.v.shape}')
# 给输入后发现神经元的电压的shape和输入的数量一致

if_layer.reset()
# 输入新的独立样本前需要调用

# IF神经元的充电过程
# spikingjelly进行离散时间仿真，使用差分形式：V[t] = V[t-1] + X[t]
def neuronal_charge(self,x:torch.Tensor):
    # self：当前神经元对象 x:当前时间步输入，且是pytorch张量
    self.v+=x

# 神经元的放电过程
# 需要判断膜电位是否达到阈值
def neuronal_fire(self):
    return self.surrogate_function(self.x-self.v_threshold)
    # surrogate_function在前向传播时是阶跃函数

# 神经元的复位过程
# 神经元发放脉冲后需要降低膜电位，包括硬复位和软复位
def neuronal_reset(self,spike):
    spike_d = spike.detach() if self.detach_reset else spike
    # 是否切断复位过程的梯度
    if self.v_reset is None:
        self.v=self.v-self.v_threshold*spike_d