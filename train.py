import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tinycudann as tcnn




print(torch.cuda.is_available())


# import os, torch
# print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
# print("torch:", torch.__version__)
# print("torch built with cuda:", torch.version.cuda)
# print("is_available:", torch.cuda.is_available())
# print("device_count:", torch.cuda.device_count())