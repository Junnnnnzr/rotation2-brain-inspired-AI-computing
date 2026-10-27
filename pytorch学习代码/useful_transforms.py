from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("image.jpg")
print(img)

# ToTensor  把图片转换成tensor类型
trans_totensor=transforms.ToTensor() # 选择transforms里的一个class进行创建具体的工具
img_tensor=trans_totensor(img) # 得到输出
writer.add_image("Tensor",img_tensor)

# Normalize 归一化
print(img_tensor[0][0][0]) # 先输出图片的第一层第一行第一列
trans_norm=transforms.Normalize([1, 3, 5], [3, 2, 1]) # 均值和标准差
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,1)

# Resize
print(img.size)
trans_resize=transforms.Resize((512,512))
# img PIL-> resize -> img_resize PIL
img_resize=trans_resize(img)
# img_resize PIL -> totensor ->img_resize tensor
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)

# Compose - resize  compose中数据需要的是transforms类型列表,作用是把多个数据转换操作组合到一起
trans_resize_2=transforms.Resize(512) # 保持原图的宽高比例，将较短边缩放到512
# PIL -> PIL -> tensor
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

# RandomCrop
trans_random=transforms.RandomCrop(512) # 得到512x512
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10): # 随机裁剪十次
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()