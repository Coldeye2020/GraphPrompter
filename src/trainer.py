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
from torch_geometric.loader import DataLoader
import sys
from ..utils.dataset import TUDataset_aug, OGBDataset_aug
from ..utils.utils import setup_seed, evaluate_embedding, arg_parse
from baseline import DGCL
import optuna
import time


def objective(trial, model="DGCL", DS="MUTAG"):
    # hyperparameters
    args = arg_parse(DS)
    setup_seed(args.seed)
    accuracies = {'result': [], 'result_val': []}
    epochs = args.epoch
    log_interval = args.log_interval
    batch_size = args.batch
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    args.hidden_dim = trial.suggest_int("hidden_dim", 128, 512, log=True)
    args.num_latent_factors = trial.suggest_int("num_latent", 3, 9, step=2)
    args.num_gc_layers = trial.suggest_int("num_gc_layer", 2, 4)
    args.pool = trial.suggest_categorical("pooling", ["max", "mean"])
    args.drop_ratio = trial.suggest_float("drop_rate", 0.2, 0.8, step=0.3)
    args.hidden_dim = int(args.hidden_dim // args.num_latent_factors) * args.num_latent_factors


    # Dataloader
    path = '../data/'
    if DS in ["HIV", "BBBP", "CLINTOX", "TOX21", "SIDER"]:
        dataset = OGBDataset_aug(root=path, name=DS, aug=args.aug)
        dataset_eval = OGBDataset_aug(root=path, name=DS, aug='none')
    elif DS in ["MUTAG", "IMDB-MULTI", "PROTEINS", "PTC_MR"]:
        dataset = TUDataset_aug(root=path, name=DS, aug=args.aug)
        dataset_eval = TUDataset_aug(root=path, name=DS, aug='none')
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, num_workers=args.num_workers)

    # Device, Model & Optimizer
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if model == "DGCL":
        model = DGCL(
            num_features=dataset_num_features,
            hidden_dim=args.hidden_dim,
            num_layer=args.num_gc_layers,
            device=device,
            args=args
        ).to(device)
    else:
        pass
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # model.eval()
    # emb, y = model.encoder.get_embeddings(dataloader_eval)

    for epoch in range(1, epochs + 1):
        # Training Process
        loss_all = 0
        model.train()
        t_total = 0
        t_read = 0
        t_aug = 0
        t_current = time.time()
        for data in dataloader:
            data, data_aug = data
            optimizer.zero_grad()
            node_num, _ = data.x.size()
            data = data.to(device)
            t_1 = time.time()
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)
            t_2 = time.time()
            edge_idx = data_aug.edge_index.numpy()
            _, edge_num = edge_idx.shape
            idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
            node_num_aug = len(idx_not_missing)
            data_aug.x = data_aug.x[idx_not_missing]
            data_aug.batch = data.batch[idx_not_missing]
            idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
            edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                        not edge_idx[0, n] == edge_idx[1, n]]
            data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
            data_aug = data_aug.to(device)
            t_3 = time.time()
            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            loss = model.loss_cal(x, x_aug)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            t_total += time.time() - t_current
            t_read += t_1 - t_current
            t_aug += t_3 - t_2
            t_current = time.time()
        aug_r = t_aug / t_total
        read_r = t_read / t_total
        train_r = 1 - aug_r - read_r
        print('Training loss: {:.3f}, Time comsumption: aug={:.2f}, read={:.2f}, train_r={:.2f}'.format(loss_all / len(dataloader), aug_r, read_r, train_r))
        if log_interval > 0 and epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            result, result_val = evaluate_embedding(emb, y)
            accuracies['result'].append(result)
            accuracies['result_val'].append(result_val)
            trial.report(result_val, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    print(accuracies)
    return accuracies["result_val"][-1]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    