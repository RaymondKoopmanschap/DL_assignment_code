import numpy as np
from modules import *
from train_mlp_pytorch import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import cifar10_utils

a = torch.zeros(5)
print(a.shape[0])