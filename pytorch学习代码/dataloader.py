import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
# drop_last=True:每轮取batchsize个，整除后的余数要不要保留
# shuffle=True: 取两轮数据取出来的顺序不同，而False则相同

img,target= test_data[0] # 第一张图片
print(img.shape)
print(target)

writer=SummaryWriter("dataloader")
for epoch in range(2): # 数据取两轮
    step=0
    for data in test_loader:
        imgs,targets= data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step+=1

writer.close()