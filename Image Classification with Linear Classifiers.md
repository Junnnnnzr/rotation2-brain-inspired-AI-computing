tensors  
1.collect dataset
2.train a classifier
3.evaluate the classifier on new images 

# 1.监督学习与分类

监督学习的数据由输入和标签组成：

- 输入：x，例如一张图像；
- 标签：y，例如“猫”或“狗”；
- 模型：f(x; θ)；
- 参数：θ，包括网络中的权重和偏置。

训练过程：

1. 将输入送入模型；
2. 得到预测结果；
3. 使用损失函数比较预测结果和真实标签；
4. 使用反向传播计算梯度；
5. 使用优化器更新参数；
6. 重复训练，降低损失。

完整分类流程：

收集数据 → 划分数据集 → 训练模型 → 验证和调参 → 测试模型 → 分析结果

# 2. nearest neighbor classifier
   distance metric:  L1 distance    
    ![[Pasted image 20260711170406.png|495]]



```python
#加载数据时要分成训练数据和标签，测试数据和标签
nn=NearestNeighbor()
nn.train(xtr_rows,ytr) #训练数据，实则保存数据
yte_predict=nn.predict(xte_rows) #利用测试集预测
print(f'accuracy:{np.mean(yte_predict==yte)')
#如何利用L1距离实现nn分类器
import numpy as np
class NearestNeighbor(object):
	def __init__(self):
		pass
	def train(self,x,y): #保存数据
		self.xtr=x
		self.ytr=y
	def predict(self,x)：
		num_test=x.shape[0]
		ypred=np.zeros(num_test,dtype=self.ytr.dtype) #构造全是0的预测的标签集
		for i in range(num_test):
			distances=np.sum(np.abs(self.xtr-x[i,:]),axis=1)
			minindex=np.argmin(distances)
			ypred[i]=self.ytr[minindex]
		return ypred
```



# 3.K-Nearest Neighbors     
 nn是只找最近的一张图片，容易被异常值影响；而knn找最相似的k个图片的标签，让他们进行投票，这样对异常值抵抗力更强

## 3.1用于超参数调优的验证集

从训练集里选取部分作为验证集，假的测试集
```python
xval_rows=xtr_rows[:1000,:]
yval=ytr[:1000]
xtr_rows=xtr_rows[1000:,:]
ytr=ytr[1000:]
for k in [1,3,5,...]
```
## 3.2交叉验证

将训练集分成5folds，其中4份用于训练，1份用于验证，然后循环这么取

# 4. CNN卷积神经网络

通过卷积核在图像上滑动来提取局部特征，具有以下特点：
1. 局部连接：每次只观察图像的一小块区域
2. 权重共享：同一个卷积核可以在图像不同位置使用
3. 保留空间结构：能够利用相邻像素之间的关系
4. 分层提取特征：浅层学习边缘和纹理，深层学习形状和物体语义
5. 参数较少：相比同等输入规模的全连接网络更容易训练