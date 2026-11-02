def LIF(Vmem_o,I,R=1e+5,C=5e-7,dt=0.01):
    '''
    模拟LIF模型
    :param Vmem_o: (float) 原始膜电位
    :param I:(float) 输入电流
    :param R:(float) 电阻
    :param C:(float) 电容
    :param dt:(float) 时间间隔/时间步
    :return:(float) 更新以后的膜电位
    '''
    tau= R*C # 时间常数
    Vmem=Vmem_o+(dt/tau)*(-Vmem_o+I*R)
    return Vmem



import numpy as np
import matplotlib.pyplot as plt

# 第一个问题：
# 假设脉冲神经元在某一时刻的膜电压为1.5V，在不受任何刺激的情况下，他的膜电压在1秒内该如何变化？
# 不再接收任何刺激说明之后的输入电流全为0
time= np.arange(0,1,0.01) # 生成从0-1秒的时间序列，间隔为0.01.不包含终点1
Vmem=[1.5] # 创建列表记录每个时间点的膜电位
for i in range(0,len(time)-1):
    Vmem.append(LIF(Vmem[i],I=0))

plt.figure(figsize=(10,5)) # 创建一张新图，规定大小
plt.plot(time,Vmem,label='Membrane Potential(V)') # 绘制曲线，x轴，y轴，曲线名称
plt.title('LIF Neuron Membrane Potential Without Stimulation') # 图片标题
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.legend() # 显示图例
plt.grid(True) # 显示网格线
plt.show()

# 第二个问题：
# 当一个神经元的静息电位为0时，此时突然给脉冲神经元一个脉冲刺激，且权重为0.1，请问在刺激后的下一个delta t时，膜电位变为多少？
# time= np.arange(0,1,0.01)
# Vmem=[0] # 静息电位为0
# # 神经元的膜电位具有状态和“记忆”，单次刺激的影响不会立刻消失，而是通过漏电逐渐衰减
# for i in range(0,len(time)-1):
#     if i==50:
#         Vmem.append(LIF(Vmem[i],I=0.1)) # 在这个时刻给一个脉冲刺激
#     else:
#         Vmem.append(LIF(Vmem[i],I=0)) # 其他时刻没有刺激，完全看膜电位的记忆



# 第三个问题：
# 权重为0.1，当一个神经元的静息电位为0时，此时突然给脉冲神经元一个脉冲刺激，等待0.1秒后，再给一个脉冲刺激，膜电位在此刻的下一个delta t变为多少？
# time= np.arange(0,1,0.01)
# Vmem=[0] # 静息电位为0
#
# for i in range(0,len(time)-1):
#     if i==50 or i==60:
#         Vmem.append(LIF(Vmem[i],I=0.1)) # 在这个时刻给一个脉冲刺激
#     else:
#         Vmem.append(LIF(Vmem[i],I=0)) # 其他时刻没有刺激，完全看膜电位的记忆



# 第四个问题：
# 当一个神经元的静息电位为0时，此时突然给脉冲神经元一连串脉冲刺激且不停下，权重为1，我们知道此时每一个脉冲产生的膜电压的增长会无限叠加，那么请问膜电位会一直无限增大吗？
# Vmem = [0]
# for i in range(0, len(time)-1):
#  if i > 30:
#     Vmem.append(LIF(Vmem[i], I=0.1))
#  else:
#     Vmem.append(LIF(Vmem[i], I=0))