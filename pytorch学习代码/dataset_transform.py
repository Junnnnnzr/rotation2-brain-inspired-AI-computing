import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set=torchvision.datasets.CIFAR10(root="./dataset",transform=dataset_transform,train=True,download=True) # root：数据下载的根目录 train=True：下载训练集 download=True：要下载
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True) # train=False：下载测试集

# print(test_set[0])
# print(test_set.classes)
#
# img,target=test_set[0]
# print(img)
# print(target) # 表示类别
# img.show()

# print(test_set[0])

writer=SummaryWriter("p10")
for i in range(10):
    img,target=train_set[i]
    writer.add_image("test_set",img,i)

writer.close()