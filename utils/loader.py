import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

ogb_list = ["hiv", "bbbp", "clintox", "tox21", "sider"]
tu_list = ["mutag", "dd", "proteins", "enzymes", "collab", "imdb-binary", "reddit-binary"]


def get_split_loader(name, root, train_ratio, val_ratio, batch_size, num_workers=4, only_trian=False):
    if name.lower() in ogb_list:
        dataset = PygGraphPropPredDataset(name="ogbg-mol" + name.lower(), root=root)
        split_idx = dataset.get_idx_split()
    elif name.lower() in tu_list:
        dataset = TUDataset(name=name.upper(), root=root)
        num_graphs = len(dataset)
        perm = torch.randperm(num_graphs)
        train_num = int(num_graphs * train_ratio)
        val_num = int(num_graphs * val_ratio)
        split_idx = {'train': perm[:train_num].long(), 'valid': perm[train_num:train_num+val_num].long(), 'test': perm[train_num+val_num:].long()}
    else:
        raise NameError
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if len(split_idx["train"]) > 3500:
        perm = torch.randperm(split_idx['train'].size(0))
        k = int(len(split_idx['train']) * 0.1)
        idx = perm[:k]
        selected_idx = split_idx['train'][idx]
        part_train_loader = DataLoader(dataset[selected_idx], batch_size=batch_size, shuffle=True)
    else:
        part_train_loader = train_loader
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader, train_loader, valid_loader, test_loader, part_train_loader
