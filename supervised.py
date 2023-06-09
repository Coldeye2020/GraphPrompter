import torch
from torch_geometric.data import DataLoader
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
import numpy as np
# from model.model import DGCL, GraphCL
import GCL.losses as L
from GCL.models import DualBranchContrast
import optuna
import time
from sklearn.model_selection import StratifiedKFold
from utils.dataset import TUDataset_aug, OGBDataset_aug
from utils.utils import setup_seed, evaluate_embedding, arg_parse
from utils.loader import get_split_loader
from GCL.augmentors import *
from utils.scheduler import CosineDecayScheduler
from ogb.graphproppred import Evaluator
import argparse
from model.model import *
from model.encoder import *
import time
import torch.optim as optim
from torch_geometric.utils import degree

import warnings
warnings.filterwarnings('ignore')


ogb_list = ["hiv", "bbbp", "clintox", "tox21", "sider"]
tu_list = ["mutag", "dd", "proteins", "enzymes", "collab", "imdb-binary", "reddit-binary"]




def create_objective(args):
    def objective(trial):
        # prerequities, recoder
        setup_seed(args.seed)
        accuracies = {'result': [], 'result_val': []}
        if args.dataset.lower() in ogb_list:
            evaluator = Evaluator("ogbg-mol" + args.dataset.lower())
        
        
        # Hyperparameters:
        if args.search == True:
            print("G")
            args.lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            args.h_dim = trial.suggest_int("hidden_dim", 32, 200, log=True)
            # args.num_gc_layers = trial.suggest_int("num_gc_layer", 2, 4)
            args.drop_ratio = trial.suggest_float("drop_rate", 0.3, 0.7)
            args.n_layers = trial.suggest_int("n_layers", 2, 4)
            args.weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-3, 1e-4, 1e-5])
        
        # Dataset: dataloader, train_loader, val_loader, test_loader
        # Augmentation: aug1, aug2
        dataloader, non_test_loader, train_loader, val_loader, test_loader, part_train_loader, train_dataset = get_split_loader(name=args.dataset, root=args.root, train_ratio=args.train_ratio, val_ratio=args.val_ratio, batch_size=args.batch_size, num_workers=args.num_workers)
        if args.dataset.lower() not in ogb_list:
            try:
                args.in_dim = next(iter(dataloader)).x.shape[-1]
            except:
                args.in_dim = 1
        args.num_classes = train_dataset.num_classes


        try:
            deg = torch.load("HIV_deg.pt")
        except:
            # Compute the maximum in-degree in the training data.
            t_s = time.time()
            max_degree = -1
            for data in train_dataset:
                d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                max_degree = max(max_degree, int(d.max()))

            # Compute the in-degree histogram tensor
            deg = torch.zeros(max_degree + 1, dtype=torch.long)
            for data in train_dataset:
                d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
                deg += torch.bincount(d, minlength=deg.numel())
            print("Degree Time comsumption: {:2f}".format(time.time() - t_s))
            torch.save(deg, "HIV_deg.pt")
        

        # Device & Encoder & Contrast_model
        device = torch.device(args.device) 
        encoder = eval(args.encoder)(in_dim=args.in_dim, h_dim=args.h_dim, n_layers=args.n_layers, drop_ratio=args.drop_ratio, deg=deg, args=args).to(device)


        # Loss & Optimizer
        model = eval(args.model)(encoder, args).to(device)

        # parameters check
        total_param = 0
        print("MODEL DETAILS:")
        # print(model)
        for param in model.encoder.parameters():
            # print(param.data.size())
            total_param += np.prod(list(param.data.size()))
        print('PNA Total parameters:', total_param)
        
        if args.use_scheduler:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5,
                                                     patience=20,
                                                     verbose=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        

        # Training:
        all_results = []
        best_epoch = 0
        best_val, best_test = 0, 0
        epoch_val_ROC, epoch_test_ROC = 0, 0
        for epoch in range(1, args.epochs+1):
            model.train()
            t_0 = time.time()
            total_num = 0
            epoch_train_loss = 0
            epoch_train_ROC = 0
            list_preds = []
            list_labels = []
            optimizer.zero_grad()

            # if args.use_scheduler:
            #     lr = scheduler.get(epoch)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
        
            for batch in train_loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                z, pred = model(batch.x, batch.edge_index, batch.batch)
                loss = model.cal_loss(pred, batch.y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.detach().item() * batch.num_graphs
                total_num += batch.num_graphs
                list_preds.append(pred.detach())
                list_labels.append(batch.y.detach())

            epoch_train_ROC = evaluator.eval({'y_pred': torch.cat(list_preds)[:,1:2],
                                       'y_true': torch.cat(list_labels)})[evaluator.eval_metric]
            epoch_train_loss = epoch_train_loss / total_num
            t_1 = time.time()
            # torch.save(torch.cat(list_labels), "test_y.pt")
            # torch.save(torch.cat(list_preds), "train_pred.pt")
            # print("Epoch: {:03d}, {}: {:.3f}(train), loss: {:.2f} || time consumption: {:.2f}s".format(epoch, evaluator.eval_metric, epoch_train_ROC, epoch_train_loss, t_1 - t_0))

           
            # Evaluating
            if args.log_interval > 0 and epoch % args.log_interval == 0:
                model.eval()
                if args.dataset.lower() in ogb_list:
                    val_pred, val_y, epoch_val_loss = model.get_pred(val_loader, device)
                    test_pred, test_y, epoch_test_loss = model.get_pred(test_loader, device)
                    # torch.save(val_pred, "val_pred.pt")
                    # torch.save(val_y, "val_y.pt")
                    # torch.save(test_pred, "test_pred.pt")
                    # torch.save(test_y, "test_y.pt")
                    epoch_val_ROC = evaluator.eval({'y_true': val_y, 'y_pred': val_pred[:,1:2]})[evaluator.eval_metric]
                    epoch_test_ROC = evaluator.eval({'y_true': test_y, 'y_pred': test_pred[:,1:2]})[evaluator.eval_metric]
                    t_2 = time.time()
                    print("Epoch: {:03d}, {}: {:.3f}(train), {:.3f}(valid), {:.3f}(test), loss: {:.3f}(train), {:.3f}(valid), {:.3f}(test), lr:{:.5f} || time consumption: {:.2f}s(train), {:.2f}s(val & test)".format(epoch, evaluator.eval_metric, epoch_train_ROC, epoch_val_ROC, epoch_test_ROC, epoch_train_loss, epoch_val_loss, epoch_test_loss, optimizer.param_groups[0]['lr'], t_1 - t_0, t_2 - t_1))
                else:
                    # x, y = model.encoder.get_embedding(dataloader, device)
                    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
                    # accuracies = []
                    # # maf1s = []
                    # # mif1s = []
                    # for train_idx, test_idx in kf.split(x, y):

                    #     split = {'train': train_idx, 'test': test_idx}
                    #     result = evaluator.evaluate(x, y, split)
                    #     accuracies.append(result["accuracy"])
                    #     # maf1s.append(result["macro_f1"])
                    #     # mif1s.append(result["micro_f1"])
                    # # results = {'micro_f1': np.mean(mif1s), 'macro_f1': np.mean(maf1s), 'accuracy': np.mean(accuracies)}
                    # test_score = np.mean(accuracies)
                    # t_2 = time.time()
                    # print("Epoch: {:03d}, acc: {:.2f}, time consumption: {:.2f}s".format(epoch, test_score, t_2 - t_1))
                    pass
                # print("Epoch: {:03d}, acc: {:.2f}, time consumption: {:.2f}s".format(epoch, test_score, t_2 - t_1))
                if best_val < epoch_val_ROC:
                    best_val = epoch_val_ROC
                    best_test = epoch_test_ROC
                trial.report(epoch_test_ROC, epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            scheduler.step(-epoch_val_ROC.item())
        return best_test

    return objective



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCL Arguments.')
    parser.add_argument('--dataset', type=str, default="HIV")
    parser.add_argument('--model', type=str, choices=["GraphCL, PNA"], default="PNA")
    parser.add_argument('--encoder', type=str, choices=["GConv, PNAConv"], default="PNAConv")
    parser.add_argument("--device", type=int, default=3)
    args = parser.parse_args()
    args = arg_parse(parser, args.model, args.encoder, args.dataset)
    

    # if args.search == True:
    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective(args), n_trials=100)
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
    # else:
    #     pass