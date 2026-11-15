# 检查 Python、PyTorch、CUDA、OpenCV 和 SpikingJelly 环境

import sys

import cv2
import matplotlib
import torch
import spikingjelly
from PIL import Image # 保存GIF动画


print("===== Environment Check =====")

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("OpenCV version:", cv2.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Pillow module:", Image.__file__)
print("SpikingJelly path:", spikingjelly.__file__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

print("===== Check Finished =====")