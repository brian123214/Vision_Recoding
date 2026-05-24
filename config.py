import gc
import os
import torch
import numpy as np
import random
import cv2
import json
import glob
import copy
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
from typing import Dict, List, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, Gemma3ForConditionalGeneration, AutoProcessor
from shape_generator import ShapeGenerator
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


GRID_SIZE = 4
X_FACTOR = 4
NUM_SHAPES = 3
COLOR_LST = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
SHAPE_LST = ['triangle', 'circle', 'square', 'star', 'heart', 'cross']
model = None
processor = None
generator = None
steered_gen = None
MODEL_TYPE = None
IMAGE_START_TOKEN = None
IMAGE_END_TOKEN = None
PATCH_SIZE = None
tokenizer = None
