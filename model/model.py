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
from torch_geometric.nn import InstanceNorm
from torch_sparse import transpose
from torch_geometric.utils import is_undirected
from utils.utils import reorder_like





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


class PNA(nn.Module):
    def __init__(self, encoder, args):
        super(PNA, self).__init__()
        self.encoder = encoder
        self.project = MLPReadout(args.h_dim, args.num_classes)
    
    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        z, g = self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_atten=edge_atten, batch=batch)
        pred = self.project(g)
        pred = F.log_softmax(pred)
        return z, pred

    def cal_loss(self, pred, labels):
        return F.nll_loss(pred, labels[:,0])
    
    @torch.no_grad()
    def get_pred(self, dataloader, device):
        x, y = [], []
        total_num = 0
        epoch_loss = 0
        for data in dataloader:
            data = data.to(device)
            if data.x is None:
                print("G")
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

            # Get embedding
            _, pred = self.forward(x=data.x, edge_index=data.edge_index, batch=data.batch)
            loss = self.cal_loss(pred, data.y)
            epoch_loss += loss.detach().item() * data.num_graphs
            total_num += data.num_graphs
            x.append(pred)
            y.append(data.y)

        
        x = torch.cat(x, dim=0).cpu().numpy()
        y = torch.cat(y, dim=0).cpu().numpy()

        return x, y, epoch_loss / total_num




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



class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class BatchMLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


class EdgePrompter(nn.Module):
    def __init__(self, hidden_size, learn_edge_att):
        super().__init__()
        self.learn_edge_att = learn_edge_att

        if self.learn_edge_att:
            self.feature_extractor = BatchMLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=0.5)
        else:
            self.feature_extractor = BatchMLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=0.5)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class GraphPrompter(nn.Module):

    def __init__(self, encoder, n_prompters=2, learn_edge_att=True, args=None):
        super().__init__()
        self.encoder = encoder
        self.e_prompters = nn.ModuleList()
        self.n_prompters = n_prompters
        # self.device = next(self.parameters()).device
        self.learn_edge_att = learn_edge_att
        self.project = MLPReadout(args.h_dim * n_prompters, args.num_classes)
        # self.final_r = final_r
        # self.decay_interval = decay_interval
        # self.decay_r = decay_r
        for _ in n_prompters:
            self.e_prompters.append(EdgePrompter(args.h_dim, learn_edge_att=learn_edge_att))

    # def __loss__(self, att, clf_logits, clf_labels, epoch):
    #     pred_loss = self.criterion(clf_logits, clf_labels)

    #     r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r)
    #     info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

    #     loss = pred_loss + info_loss
    #     loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
    #     return loss, loss_dict

    def forward(self, x, edge_index, batch, edge_attr=None):
        node_emb, _ = self.encoder(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, edge_atten=None)
        edge_atts = []
        g_embs = []

        for i, e_p in enumerate(self.e_prompters):
            att = e_p(node_emb, edge_index, batch)

            if self.learn_edge_att:
                if is_undirected(edge_index):
                    trans_idx, trans_val = transpose(edge_index, att, None, None, coalesced=False)
                    trans_val_perm = reorder_like(trans_idx, edge_index, trans_val)
                    edge_att = (att + trans_val_perm) / 2
                else:
                    edge_att = att
            else:
                edge_att = self.lift_node_att_to_edge_att(edge_att, edge_index)
            
            edge_atts.append(edge_att)
        
        
        for edge_att in edge_atts:
            _, g_emb = self.encoder(x, edge_index, batch, edge_attr=edge_attr, edge_atten=edge_att)
            g_embs.append(g_emb)
        
        post_g_emb = torch.cat(g_embs, dim=-1)
        pred = self.project(post_g_emb)
        pred = F.log_softmax(pred)
        
        return torch.cat([i.unsqueeze(1) for i in g_embs], dim=1), pred, edge_atts
    

    def cal_loss(self, data):
        post_g_emb, pred, edge_att = self.forward(data.x, data.edge_index, data.batch, data.edge_attr)
        s_loss = F.nll_loss(pred, data.y[:,0])
        d_loss = 
        return s_loss, d_loss 
    
    @torch.no_grad()
    def get_pred(self, dataloader, device):
        x, y = [], []
        total_num = 0
        epoch_loss = 0
        for data in dataloader:
            data = data.to(device)
            if data.x is None:
                print("G")
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

            # Get embedding
            _, pred = self.forward(x=data.x, edge_index=data.edge_index, batch=data.batch)
            loss = self.cal_loss(pred, data.y)
            epoch_loss += loss.detach().item() * data.num_graphs
            total_num += data.num_graphs
            x.append(pred)
            y.append(data.y)
        
        x = torch.cat(x, dim=0).cpu().numpy()
        y = torch.cat(y, dim=0).cpu().numpy()

        return x, y, epoch_loss / total_num

    # @staticmethod
    # def sampling(att_log_logit, training):
    #     temp = 1
    #     if training:
    #         random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
    #         random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
    #         att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
    #     else:
    #         att_bern = (att_log_logit).sigmoid()
    #     return att_bern

    # @staticmethod
    # def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
    #     r = init_r - current_epoch // decay_interval * decay_r
    #     if r < final_r:
    #         r = final_r
    #     return r

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att