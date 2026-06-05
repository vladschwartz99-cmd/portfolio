# Стандартные библиотеки Python
import os
import random
import re
import xml.etree.ElementTree as ET

# Сторонние библиотеки для работы с данными и визуализации
import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Библиотеки PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
