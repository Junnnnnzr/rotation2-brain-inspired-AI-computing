import torch
import torchvision

vgg16=torchvision.models.vgg16(weights=None)
# 保存方式1
torch.save(vgg16,"vgg16_method1.pth") # 保存网络模型结构和参数

# 保存方法2 （官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth") # 将其中的参数保存成字典