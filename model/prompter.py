import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import numpy as np
from copy import deepcopy
from itertools import repeat
from torch_geometric.data import DataLoader, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader



class Edgeprompt(nn.Module):
    def __init__(self, in_dim, h_dim, n_layers, activation="relu", args=None):
        super(Edgeprompt, self).__init__()