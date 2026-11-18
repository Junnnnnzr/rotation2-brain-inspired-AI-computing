# SNN需要模拟时间变化，当前输出会受到上一时刻膜电位和当前输出的影响

# 观察代码的保存
import torch
from spikingjelly.activation_based import neuron

net_s=neuron.IFNode(step_mode='s') # net_s保存这个神经元模块的变量
# v_threshold=1.0 阈值
# v_reset=0.0 发放后膜电位复位到0
# detach_reset=False
x=torch.rand([4])
print(net_s)
print(f'the initial v={net_s.v}')
# net_s.v 保存脉冲神经元内部保存的膜电位状态，在初始化后v会被设置为0
y = net_s(x)
print(f'x={x}')
print(f'y={y}')
print(f'v={net_s.v}')

# 若我们给与一个新的输入，需要先清除指定的一个神经元之前的状态，恢复到初始化状态
net_s.reset()


from spikingjelly.activation_based import functional
for x, label in tqdm(train_data_loader):
    optimizer.zero_grad()
    y = net(x)
    loss = criterion(y, label)
    loss.backward()
    optimizer.step()

    functional.reset_net(net)
    # 当前批次完成后，重置整个网络中的所有有状态的模块

