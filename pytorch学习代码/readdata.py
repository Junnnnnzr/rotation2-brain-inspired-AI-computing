from networkx.algorithms.threshold import random_threshold_sequence
from torch.utils.data import Dataset
from PIL import Image
import os

class mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir # dataset/train
        self.label_dir = label_dir # ants bees
        self.path=os.path.join(self.root_dir,self.label_dir) # 两者合并
        self.img_path=os.listdir(self.path) # 所有图片路径的列表

    def __getitem__(self,idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name) # 得到一张图片的相对路径
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir="dataset/train"
ants_label_dir="ants"
bees_label_dir="bees"
ants_dataset=mydata(root_dir,ants_label_dir)
bees_dataset=mydata(root_dir,bees_label_dir)

train_dataset=ants_dataset+bees_dataset # 数据集的拼接