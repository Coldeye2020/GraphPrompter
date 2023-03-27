import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import os
import os.path as osp
import shutil
import numpy as np
import pickle
import yaml
from copy import deepcopy
from itertools import repeat
from torch_geometric.data import DataLoader, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import sys
from ..utils.dataset import TUDataset_aug, OGBDataset_aug
from utils import setup_seed, evaluate_embedding


class DisenEncoder(torch.nn.Module):
    def __init__(self, num_features, emb_dim, num_layer, K, head_layers, if_proj_head=False, drop_ratio=0.0,
                 graph_pooling='add', JK='last', residual=False, device=None, args=None):
        super(DisenEncoder, self).__init__()
        self.args = args
        self.device = device
        self.num_features = num_features
        self.K = K
        self.d = emb_dim // self.K
        self.num_layer = num_layer
        self.head_layers = head_layers
        self.gc_layers = self.num_layer - self.head_layers
        self.if_proj_head = if_proj_head
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        if self.graph_pooling == "sum" or self.graph_pooling == 'add':
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        self.JK = JK
        if JK == 'last':
            pass
        elif JK == 'sum':
            self.JK_proj = Linear(self.gc_layers * emb_dim, emb_dim)
        else:
            assert False
        self.residual = residual
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.disen_convs = torch.nn.ModuleList()
        self.disen_bns = torch.nn.ModuleList()

        for i in range(self.gc_layers):
            if i == 0:
                nn = Sequential(Linear(num_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            else:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(emb_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        for i in range(self.K):
            for j in range(self.head_layers):
                if j == 0:
                    nn = Sequential(Linear(emb_dim, self.d), ReLU(), Linear(self.d, self.d))
                else:
                    nn = Sequential(Linear(self.d, self.d), ReLU(), Linear(self.d, self.d))
                conv = GINConv(nn)
                bn = torch.nn.BatchNorm1d(self.d)

                self.disen_convs.append(conv)
                self.disen_bns.append(bn)

        self.proj_heads = torch.nn.ModuleList()
        for i in range(self.K):
            nn = Sequential(Linear(self.d, self.d), ReLU(inplace=True), Linear(self.d, self.d))
            self.proj_heads.append(nn)

    def _normal_conv(self, x, edge_index, batch):
        xs = []
        for i in range(self.gc_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i == self.gc_layers - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            if self.residual and i > 0:
                x += xs[i - 1]
            xs.append(x)
        if self.JK == 'last':
            return xs[-1]
        elif self.JK == 'sum':
            return self.JK_proj(torch.cat(xs, dim=-1))

    def _disen_conv(self, x, edge_index, batch):
        x_proj_list = []
        x_proj_pool_list = []
        for i in range(self.K):
            x_proj = x
            for j in range(self.head_layers):
                tmp_index = i * self.head_layers + j
                x_proj = self.disen_convs[tmp_index](x_proj, edge_index)
                x_proj = self.disen_bns[tmp_index](x_proj)
                if j != self.head_layers - 1:
                    x_proj = F.relu(x_proj)
            x_proj_list.append(x_proj)
            x_proj_pool_list.append(self.pool(x_proj, batch))
        if self.if_proj_head:
            x_proj_pool_list = self._proj_head(x_proj_pool_list)
        x_graph_multi = torch.stack(x_proj_pool_list)
        x_node_multi = torch.stack(x_proj_list)
        x_graph_multi = x_graph_multi.permute(1, 0, 2).contiguous()
        x_node_multi = x_node_multi.permute(1, 0, 2).contiguous()
        return x_graph_multi, x_node_multi

    def _proj_head(self, x_proj_pool_list):
        ret = []
        for k in range(self.K):
            x_graph_proj = self.proj_heads[k](x_proj_pool_list[k])
            ret.append(x_graph_proj)
        return ret

    def forward(self, x, edge_index, batch, device="mps"):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        h_node = self._normal_conv(x, edge_index, batch)
        h_graph_multi, h_node_multi = self._disen_conv(h_node, edge_index, batch)
        return h_graph_multi, h_node_multi

    def get_embeddings(self, loader):
        device = self.device
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch)
                B, K, d = x.size()
                x = x.view(B, K * d)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

class DGCL(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layer, device, args):
        super(DGCL, self).__init__()

        self.args = args
        self.device = device
        self.T = self.args.tau
        self.K = args.num_latent_factors
        self.embedding_dim = hidden_dim
        self.d = self.embedding_dim // self.K

        self.center_v = torch.rand((self.K, self.d), requires_grad=True).to(device)

        self.encoder = DisenEncoder(
            num_features=num_features,
            emb_dim=hidden_dim,
            num_layer=num_layer,
            K=args.num_latent_factors,
            head_layers=args.head_layers,
            device=device,
            args=args,
            if_proj_head=args.proj > 0,
            drop_ratio=args.drop_ratio,
            graph_pooling=args.pool,
            JK=args.JK,
            residual=args.residual > 0
        )

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        z_graph, _ = self.encoder(x, edge_index, batch)
        return z_graph

    def loss_cal(self, x, x_aug):
        T = self.T
        T_c = 0.2
        B, H, d = x.size()
        ck = F.normalize(self.center_v)
        p_k_x_ = torch.einsum('bkd,kd->bk', F.normalize(x, dim=-1), ck)
        p_k_x = F.softmax(p_k_x_ / T_c, dim=-1)
        x_abs = x.norm(dim=-1)
        x_aug_abs = x_aug.norm(dim=-1)
        x = torch.reshape(x, (B * H, d))
        x_aug = torch.reshape(x_aug, (B * H, d))
        x_abs = torch.squeeze(torch.reshape(x_abs, (B * H, 1)), 1)
        x_aug_abs = torch.squeeze(torch.reshape(x_aug_abs, (B * H, 1)), 1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (1e-8 + torch.einsum('i,j->ij', x_abs, x_aug_abs))
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(B * H), range(B * H)]
        score = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim)
        p_y_xk = score.view(B, H)
        q_k = torch.einsum('bk,bk->bk', p_k_x, p_y_xk)
        q_k = F.normalize(q_k, dim=-1)
        elbo = q_k * (torch.log(p_k_x) + torch.log(p_y_xk) - torch.log(q_k))
        loss = - elbo.view(-1).mean()
        return loss