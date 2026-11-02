# 向LIF中引入阈值：
# 如果脉冲神经元的膜电位达到了阈值，那么他就释放一个脉冲，随后膜电位会被重置到一个特定的电位值

def LIF(Vmem_o, thr, I, R=1e+5, C=5e-7, dt=0.01):
 tau = R*C
 Vmem = Vmem_o + (dt / tau) * (-Vmem_o + I * R)
 if Vmem > thr: # 每得到一个膜电位值就与阈值比较，如果超过了阈值，那么就释放一个脉冲，并把膜电位重置到0
    spike = 1
    Vmem = 0
 else: spike = 0
 return Vmem, spike

import numpy as np
import matplotlib.pyplot as plt

vmem_list=[]
spike_list=[]
input=[0]*10+[0.01]*40
Vmem=0
for step in range(0,len(input)):
    Vmem,spike=LIF(Vmem_o=Vmem,thr=500,I=input[step],R=1e+5,C=5e-7)
    vmem_list.append(Vmem)
    spike_list.append(spike)

fig,axs=plt.subplots(2,1,figsize=(10,8),sharex=True)
# 创建一张包含两个子图的图片（2行1列），共享一个横坐标。fig代表整张图片，axs是包含两个子图的对象
axs[0].plot(np.linspace(0,0.49,50),vmem_list,'-b',label='Membrane,Potential')
# 横坐标从0-0.49之间均匀生成50个时间点。-使用实线。b代表蓝色。
axs[0].set_title('Membrane Potential Over Time')
axs[0].set_ylabel('Potential(V)')
axs[0].grid(True)

axs[1].bar(np.linspace(0,0.49,50),spike_list,width=0.001,color='r',label='Spikes')
# 创建柱状图.每个脉冲柱子的宽度为0.001。r代表红色。
axs[1].set_title('Spikes Events')
axs[1].set_ylabel('Spike Event')
axs[1].set_ylim(0,1.5)
# 把纵坐标范围固定在0-1.5之间
axs[1].grid(True)

plt.tight_layout()
# 自动调整两个子图之间的距离，避免标题，坐标文字互相遮挡
plt.show()
