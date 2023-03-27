import torch
from aug import *
from copy import deepcopy
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset


class OGBDataset_aug(PygGraphPropPredDataset):
    def __init__(self, name, root, aug=None, transform=None, pre_transform=None):
        super(OGBDataset_aug, self).__init__(name="ogbg-mol" + name.lower(), root=root, transform=transform, pre_transform=pre_transform)
        self.aug = aug
    
    def get_num_feature(self):
        data = super(OGBDataset_aug, self).get(0)
        _, num_feature = data.x.size()
        return num_feature

    def get(self, idx: int):
        data = super(PygGraphPropPredDataset, self).get(idx)

        node_num = data.edge_index.max()
        sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif self.aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif self.aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif self.aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif self.aug == 'none':
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max() + 1, 1))
        elif self.aug == 'random4':
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            print('augmentation error')
            assert False

        return data, data_aug


class TUDataset_aug(TUDataset):
    def __init__(self, name, root, aug=None, transform=None, pre_transform=None):
        super(TUDataset_aug, self).__init__(name=name, root=root, transform=transform, pre_transform=pre_transform)
        self.aug = aug
    
    def get_num_feature(self):
        data = super(TUDataset_aug, self).get(0)
        _, num_feature = data.x.size()
        return num_feature

    def get(self, idx: int):
        data = super(TUDataset_aug, self).get(idx)

        node_num = data.edge_index.max()
        sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif self.aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif self.aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif self.aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif self.aug == 'none':
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max() + 1, 1))
        elif self.aug == 'random4':
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            print('augmentation error')
            assert False

        return data, data_aug
    
