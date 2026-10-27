from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# python的用法   totensor 数据类型
# 通过transforms.ToTensor看两个问题

# 2.为什么我们需要Tensor数据类型

img_path="dataset/train/ants/0013035.jpg"
img=Image.open(img_path)

writer=SummaryWriter("logs")

# 1.transforms该如何使用
tensor_trans=transforms.ToTensor() # 选择transforms里的一个class进行创建具体的工具
tensor_img=tensor_trans(img) # 得到输出

writer.add_image("Tensor_img",tensor_img)

writer.close()