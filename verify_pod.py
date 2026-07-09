import torch
import torchvision
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import PIL
import pydicom
import flwr

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("All imports successful.")