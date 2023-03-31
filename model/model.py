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


class GraphCL_E(nn.Module):
    def __init__(self, encoder, augmentor, contrast_model, args):
        super(GraphCL_E, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.contrast_model = contrast_model
        project_dim = args.h_dim * args.n_layers
        self.project = MLP(project_dim, project_dim)
    
    def forward(self, x, edge_index, edge_attr, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_attr1 = aug1(x, edge_index, edge_attr)
        x2, edge_index2, edge_attr2 = aug2(x, edge_index, edge_attr)
        z, g = self.encoder(x, edge_index, edge_attr, batch)
        z1, g1 = self.encoder(x1, edge_index1, edge_attr1, batch)
        z2, g2 = self.encoder(x2, edge_index2, edge_attr2, batch)
        return z, g, z1, z2, g1, g2

    def cal_loss(self, g, g1, g2):
        # g = self.project(g)
        g1 = self.project(g1)
        g2 = self.project(g2)
        loss = self.contrast_model(g1=g1, g2=g2)
        return loss


class GraphCL(nn.Module):
    def __init__(self, encoder, augmentor, contrast_model, args):
        super(GraphCL, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.contrast_model = contrast_model
        project_dim = args.h_dim * args.n_layers
        self.project = MLP(project_dim, project_dim)
    
    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2
    
    def cal_loss(self, g, g1, g2):
        # g = self.project(g)
        g1 = self.project(g1)
        g2 = self.project(g2)
        loss = self.contrast_model(g1=g1, g2=g2)
        return loss

        

# class DGCL(nn.Module):
#     def __init__(self, num_features, hidden_dim, num_layer, device, args):
#         super(DGCL, self).__init__()

#         self.args = args
#         self.device = device
#         self.T = self.args.tau
#         self.K = args.num_latent_factors
#         self.embedding_dim = hidden_dim
#         self.d = self.embedding_dim // self.K

#         self.center_v = torch.rand((self.K, self.d), requires_grad=True).to(device)

#         self.encoder = DisenEncoder(
#             num_features=num_features,
#             emb_dim=hidden_dim,
#             num_layer=num_layer,
#             K=args.num_latent_factors,
#             head_layers=args.head_layers,
#             device=device,
#             args=args,
#             if_proj_head=args.proj > 0,
#             drop_ratio=args.drop_ratio,
#             graph_pooling=args.pool,
#             JK=args.JK,
#             residual=args.residual > 0
#         )

#         self.init_emb()

#     def init_emb(self):
#         initrange = -1.5 / self.embedding_dim
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.fill_(0.0)

#     def forward(self, x, edge_index, batch, num_graphs):
#         if x is None:
#             x = torch.ones(batch.shape[0]).to(device)
#         z_graph, _ = self.encoder(x, edge_index, batch)
#         return z_graph

#     def loss_cal(self, x, x_aug):
#         T = self.T
#         T_c = 0.2
#         B, H, d = x.size()
#         ck = F.normalize(self.center_v)
#         p_k_x_ = torch.einsum('bkd,kd->bk', F.normalize(x, dim=-1), ck)
#         p_k_x = F.softmax(p_k_x_ / T_c, dim=-1)
#         x_abs = x.norm(dim=-1)
#         x_aug_abs = x_aug.norm(dim=-1)
#         x = torch.reshape(x, (B * H, d))
#         x_aug = torch.reshape(x_aug, (B * H, d))
#         x_abs = torch.squeeze(torch.reshape(x_abs, (B * H, 1)), 1)
#         x_aug_abs = torch.squeeze(torch.reshape(x_aug_abs, (B * H, 1)), 1)
#         sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (1e-8 + torch.einsum('i,j->ij', x_abs, x_aug_abs))
#         sim_matrix = torch.exp(sim_matrix / T)
#         pos_sim = sim_matrix[range(B * H), range(B * H)]
#         score = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim)
#         p_y_xk = score.view(B, H)
#         q_k = torch.einsum('bk,bk->bk', p_k_x, p_y_xk)
#         q_k = F.normalize(q_k, dim=-1)
#         elbo = q_k * (torch.log(p_k_x) + torch.log(p_y_xk) - torch.log(q_k))
#         loss = - elbo.view(-1).mean()
#         return loss


class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, norm_type='batch'):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(in_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.norm = nn.BatchNorm1d(h_dim)
        self.act_fn = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.norm(x)
        x = self.act_fn(x)
        x = self.layer2(x)

        return x
