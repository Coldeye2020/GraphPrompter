import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, GINEConv, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def make_gine_conv(in_dim, out_dim):
    return GINEConv(nn.Sequential(nn.Linear(in_dim, out_dim*2), torch.nn.BatchNorm1d(2*out_dim), nn.ReLU(), nn.Linear(out_dim*2, out_dim)))
def make_gin_conv(in_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
    
class GConv_E(nn.Module):
    def __init__(self, in_dim, h_dim, n_layers, drop_ratio, args=None):
        super(GConv_E, self).__init__()
        self.num_layers = n_layers
        self.dropout = drop_ratio
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.atom_encoder = AtomEncoder(h_dim)
        self.bond_encoder = BondEncoder(h_dim)
        for i in range(n_layers):
            self.layers.append(make_gine_conv(h_dim, h_dim))
            self.batch_norms.append(nn.BatchNorm1d(h_dim))
        
        
    def forward(self, x, edge_index, edge_attr, batch):
        # z = x
        z = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        zs = []
        for layer, (conv, bn) in enumerate(zip(self.layers, self.batch_norms)):
            z = conv(z, edge_index, edge_attr)
            z = bn(z)

            if layer == self.num_layers -1:
                z = F.dropout(z, self.dropout, training=self.training)
            else:
                z = F.dropout(F.relu(z), self.dropout, training=self.training)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

    @torch.no_grad()
    def get_embedding(self, dataloader, device):
        x, y = [], []
        for data in dataloader:
            data = data.to(device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

            # Get embedding
            _, g = self.forward(data.x, data.edge_index, data.edge_attr, data.batch)
        
            x.append(g)
            y.append(data.y)

        x = torch.cat(x, dim=0).cpu()
        y = torch.cat(y, dim=0).cpu()

        return x, y

class GConv(nn.Module):
    def __init__(self, in_dim, h_dim, n_layers, drop_ratio, args=None):
        super(GConv, self).__init__()
        self.n_layers = n_layers
        self.dropout = drop_ratio
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                self.layers.append(make_gin_conv(in_dim, h_dim))
            else:
                self.layers.append(make_gin_conv(h_dim, h_dim))
            self.batch_norms.append(nn.BatchNorm1d(h_dim))


    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for layer in range(self.n_layers):
            z = self.layers[layer](z, edge_index)
            z = self.batch_norms[layer](z)

            if layer == self.num_layers -1:
                z = F.dropout(z, self.dropout, training=self.training)
            else:
                z = F.dropout(F.relu(z), self.dropout, training=self.training)

            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

    @torch.no_grad()
    def get_embedding(self, dataloader, device):
        x, y = [], []
        for data in dataloader:
            data = data.to(device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)

            # Get embedding
            g, _, _, _ = self.forward(data.x, data.edge_index, data.batch)
        
            x.append(g)
            y.append(data.y)

        x = torch.cat(x, dim=0).cpu()
        y = torch.cat(y, dim=0).cpu()

        return x, y



# class DisenEncoder(torch.nn.Module):
#     def __init__(self, num_features, emb_dim, num_layer, K, head_layers, if_proj_head=False, drop_ratio=0.0,
#                  graph_pooling='add', JK='last', residual=False, device=None, args=None):
#         super(DisenEncoder, self).__init__()
#         self.args = args
#         self.device = device
#         self.num_features = num_features
#         self.K = K
#         self.d = emb_dim // self.K
#         self.num_layer = num_layer
#         self.head_layers = head_layers
#         self.gc_layers = self.num_layer - self.head_layers
#         self.if_proj_head = if_proj_head
#         self.drop_ratio = drop_ratio
#         self.graph_pooling = graph_pooling
#         if self.graph_pooling == "sum" or self.graph_pooling == 'add':
#             self.pool = global_add_pool
#         elif self.graph_pooling == "mean":
#             self.pool = global_mean_pool
#         elif self.graph_pooling == "max":
#             self.pool = global_max_pool
#         else:
#             raise ValueError("Invalid graph pooling type.")
#         self.JK = JK
#         if JK == 'last':
#             pass
#         elif JK == 'sum':
#             self.JK_proj = Linear(self.gc_layers * emb_dim, emb_dim)
#         else:
#             assert False
#         self.residual = residual
#         self.convs = torch.nn.ModuleList()
#         self.bns = torch.nn.ModuleList()
#         self.disen_convs = torch.nn.ModuleList()
#         self.disen_bns = torch.nn.ModuleList()

#         for i in range(self.gc_layers):
#             if i == 0:
#                 nn = Sequential(Linear(num_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
#             else:
#                 nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
#             conv = GINConv(nn)
#             bn = torch.nn.BatchNorm1d(emb_dim)

#             self.convs.append(conv)
#             self.bns.append(bn)

#         for i in range(self.K):
#             for j in range(self.head_layers):
#                 if j == 0:
#                     nn = Sequential(Linear(emb_dim, self.d), ReLU(), Linear(self.d, self.d))
#                 else:
#                     nn = Sequential(Linear(self.d, self.d), ReLU(), Linear(self.d, self.d))
#                 conv = GINConv(nn)
#                 bn = torch.nn.BatchNorm1d(self.d)

#                 self.disen_convs.append(conv)
#                 self.disen_bns.append(bn)

#         self.proj_heads = torch.nn.ModuleList()
#         for i in range(self.K):
#             nn = Sequential(Linear(self.d, self.d), ReLU(inplace=True), Linear(self.d, self.d))
#             self.proj_heads.append(nn)

#     def _normal_conv(self, x, edge_index, batch):
#         xs = []
#         for i in range(self.gc_layers):
#             x = self.convs[i](x, edge_index)
#             x = self.bns[i](x)
#             if i == self.gc_layers - 1:
#                 x = F.dropout(x, self.drop_ratio, training=self.training)
#             else:
#                 x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

#             if self.residual and i > 0:
#                 x += xs[i - 1]
#             xs.append(x)
#         if self.JK == 'last':
#             return xs[-1]
#         elif self.JK == 'sum':
#             return self.JK_proj(torch.cat(xs, dim=-1))

#     def _disen_conv(self, x, edge_index, batch):
#         x_proj_list = []
#         x_proj_pool_list = []
#         for i in range(self.K):
#             x_proj = x
#             for j in range(self.head_layers):
#                 tmp_index = i * self.head_layers + j
#                 x_proj = self.disen_convs[tmp_index](x_proj, edge_index)
#                 x_proj = self.disen_bns[tmp_index](x_proj)
#                 if j != self.head_layers - 1:
#                     x_proj = F.relu(x_proj)
#             x_proj_list.append(x_proj)
#             x_proj_pool_list.append(self.pool(x_proj, batch))
#         if self.if_proj_head:
#             x_proj_pool_list = self._proj_head(x_proj_pool_list)
#         x_graph_multi = torch.stack(x_proj_pool_list)
#         x_node_multi = torch.stack(x_proj_list)
#         x_graph_multi = x_graph_multi.permute(1, 0, 2).contiguous()
#         x_node_multi = x_node_multi.permute(1, 0, 2).contiguous()
#         return x_graph_multi, x_node_multi

#     def _proj_head(self, x_proj_pool_list):
#         ret = []
#         for k in range(self.K):
#             x_graph_proj = self.proj_heads[k](x_proj_pool_list[k])
#             ret.append(x_graph_proj)
#         return ret

#     def forward(self, x, edge_index, batch, device="mps"):
#         if x is None:
#             x = torch.ones((batch.shape[0], 1)).to(device)
#         h_node = self._normal_conv(x, edge_index, batch)
#         h_graph_multi, h_node_multi = self._disen_conv(h_node, edge_index, batch)
#         return h_graph_multi, h_node_multi

#     def get_embeddings(self, loader):
#         device = self.device
#         ret = []
#         y = []
#         with torch.no_grad():
#             for data in loader:
#                 data = data[0]
#                 data.to(device)
#                 x, edge_index, batch = data.x, data.edge_index, data.batch
#                 if x is None:
#                     x = torch.ones((batch.shape[0], 1)).to(device)
#                 x, _ = self.forward(x, edge_index, batch)
#                 B, K, d = x.size()
#                 x = x.view(B, K * d)
#                 ret.append(x.cpu().numpy())
#                 y.append(data.y.cpu().numpy())
#         ret = np.concatenate(ret, 0)
#         y = np.concatenate(y, 0)
#         return ret, y