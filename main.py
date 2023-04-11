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
from utils.Evaluator import SVMEvaluator, OGBEvaluator
from ogb.graphproppred import Evaluator
import argparse
from model.model import *
from model.encoder import *
import time

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
            if args.dataset.lower() == 'hiv':
                evaluator = OGBEvaluator(evaluator, base_classifier='svm', param_search=False)
            else:
                evaluator = OGBEvaluator(evaluator, base_classifier='lr', param_search=False)
        else:
            evaluator = SVMEvaluator(linear=True)
        
        
        # Hyperparameters:
        args.lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        args.h_dim = trial.suggest_int("hidden_dim", 32, 200, log=True)
        # args.num_gc_layers = trial.suggest_int("num_gc_layer", 2, 4)
        args.drop_ratio = trial.suggest_float("drop_rate", 0.3, 0.7)
        args.intraview_negs = trial.suggest_categorical('intraview_negs', [True, False])
        args.tau = trial.suggest_float('temperature', 0.2, 1.0)
        args.n_layers = trial.suggest_int("n_layers", 2, 4)
        args.weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-3, 1e-4, 1e-5])
        if args.model == "DGCL":
            args.pool = trial.suggest_categorical("pooling", ["max", "mean"])
            args.num_latent_factors = trial.suggest_int("num_latent", 3, 9, step=2)
            args.h_dim = int(args.h_dim // args.num_latent_factors) * args.num_latent_factors
        if args.model == "GPrompter":
            args.lam_d = trial.suggest_float('lam_d', 0.5, 2) # 这个超参数还是不能太确定，得看具体的去相关Loss的值来估算下
        args.aug_strength = trial.suggest_categorical('aug_strength', [0.1, 0.2, 0.3])
        
        # Dataset: dataloader, train_loader, val_loader, test_loader
        # Augmentation: aug1, aug2
        dataloader, non_test_loader, train_loader, val_loader, test_loader, part_train_loader, num_classes = get_split_loader(name=args.dataset, root=args.root, train_ratio=args.train_ratio, val_ratio=args.val_ratio, batch_size=args.batch_size, num_workers=args.num_workers)
        if args.dataset.lower() not in ogb_list:
            try:
                args.in_dim = next(iter(dataloader)).x.shape[-1]
            except:
                args.in_dim = 1

        if args.two_aug:
            aug1 = [Identity(),
                    RWSampling(num_seeds=1000, walk_length=10),
                    NodeDropping(pn=args.aug_strength),
                    FeatureMasking(pf=args.aug_strength),
                    EdgeRemoving(pe=args.aug_strength)]
            aug2 = RandomChoice([Identity(),
                    RWSampling(num_seeds=1000, walk_length=10),
                    NodeDropping(pn=args.aug_strength),
                    FeatureMasking(pf=args.aug_strength),
                    EdgeRemoving(pe=args.aug_strength)], 1)

        else:
            aug1 = Identity()
            aug2 = RandomChoice([RWSampling(num_seeds=1000, walk_length=10),
                    NodeDropping(pn=0.1),
                    FeatureMasking(pf=0.1),
                    EdgeRemoving(pe=0.1)], 1)

        # Device & Encoder & Contrast_model
        device = torch.device(args.device)
        if args.dataset.lower() in ogb_list:
            encoder = eval(args.encoder + "_E")(in_dim=args.in_dim, h_dim=args.h_dim, n_layers=args.n_layers, drop_ratio=args.drop_ratio).to(device)
        else:
            encoder = eval(args.encoder)(in_dim=args.in_dim, h_dim=args.h_dim, n_layers=args.n_layers, drop_ratio=args.drop_ratio).to(device)

        # if args.model == "":
        #     encoder = DisenEncoder(in_dim=args.in_dim, h_dim=args.h_dim, n_layers=args.n_layers, drop_ratio=args.drop_ratio, ogb=args.dataset.lower() in ogb_list).to(device) # 1. 这里多了一个参数ogb，用于确定是否要使用bondencoder和edgeencoder，另外对于OGB数据集还要考虑edge_attr。2. 在encoder里面加入get_embedding函数，用于读取dataloder，并且返回一个所有图数据在cpu上的embedding(参考self.get_embedding)
        #     # encoder = ...
        #     pass
        # elif args.model == "GPrompter":
        #     # encoder = ...
        #     pass
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=args.tau), mode='G2G', intraview_negs=args.intraview_negs).to(device)
        


        # Loss & Optimizer
        if args.dataset.lower() in ogb_list:
            model = eval(args.model + "_E")(encoder=encoder, augmentor=(aug1, aug2), contrast_model=contrast_model, args=args).to(device)
        else:
            model = eval(args.model)(encoder=encoder, augmentor=(aug1, aug2), contrast_model=contrast_model, args=args).to(device)

        # if args.model == "GCL":
        #     model = eval
        # elif args.model == "DGCL":
        #     model = eval(args.encoder + "_E")(encoder=encoder, augmentor=(aug1, aug2), contrast_model=contrast_model, args=args) # 需要包含1.利用augmentor生成不同的视图，通过encoder得到表征，2.自己的cal_loss,记得加入projection head，3.建议把project head放到model中。
        # elif args.model == "GPrompter":
        #     model = GPrompter(encoder=encoder, augmentor=(aug1, aug2), contrast_model=contrast_model, args=args)
        
        if args.use_scheduler:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = CosineDecayScheduler(max_val=args.lr, warmup_steps=args.epochs//10, total_steps=args.epochs)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        

        # Training:
        all_results = []
        best_epoch = 0
        best_val, best_test = 0, 0
        for epoch in range(1, args.epochs+1):
            # t_0 = time.time()
            model.train()
            epoch_loss = 0
            optimizer.zero_grad()

            if args.use_scheduler:
                lr = scheduler.get(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        
            for batch in non_test_loader:
                batch = batch.to(device)
                if args.dataset.lower() in ogb_list:
                    z, g, z1, z2, g1, g2 = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    z, g, z1, z2, g1, g2 = model(batch.x, batch.edge_index, batch.batch)
                if args.model == "GPrompter":
                    loss, closs, dloss = model.cal_loss(g=g, g1=g1, g2=g2) # 1.通过不同的loss，记录一下不同loss之间的大小关系，方便调节超参数 2.如果之后想加入一个GVAE用来原始数据和表征之间的互信息，则可以在loss层面作出改动
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_closs += closs.item()
                    epoch_dloss += dloss.item()
                elif args.model == "DGCL" or "GraphCL":
                    loss = model.cal_loss(g=g, g1=g1, g2=g2)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            t_1 = time.time()
            # if args.model == "GPrompter":
            #     print("Epoch: {:03d}, loss: {:.2f}, closs:{:.2f}, dloss:{:.2f}, time consumption: {:.2f}s".format(epoch, epoch_loss, epoch_closs, epoch_dloss, t_1 - t_0))
            # elif args.model == "DGCL" or "GraphCL":
            #     print("Epoch: {:03d}, loss: {:.2f}, time consumption: {:.2f}s".format(epoch, epoch_loss, t_1 - t_0))

           
            # Evaluating
            if args.log_interval > 0 and epoch % args.log_interval == 0:
                model.eval()

                if args.dataset.lower() in ogb_list:
                    train_x, train_y = model.encoder.get_embedding(part_train_loader, device)
                    val_x, val_y = model.encoder.get_embedding(val_loader, device)
                    test_x, test_y = model.encoder.get_embedding(test_loader, device)
                    train_score, val_score, test_score = evaluator.evaluate(train_x, train_y, val_x, val_y, test_x, test_y)
                    t_2 = time.time()
                    print("Epoch: {:03d}, {}: {:.2f}(train) {:.2f}(valid), {:.2f}(test) || time consumption: {:.2f}s".format(epoch, evaluator.eval_metric, train_score, val_score, test_score, t_2 - t_1))
                else:
                    x, y = model.encoder.get_embedding(dataloader, device)
                    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
                    accuracies = []
                    # maf1s = []
                    # mif1s = []
                    for train_idx, test_idx in kf.split(x, y):

                        split = {'train': train_idx, 'test': test_idx}
                        result = evaluator.evaluate(x, y, split)
                        accuracies.append(result["accuracy"])
                        # maf1s.append(result["macro_f1"])
                        # mif1s.append(result["micro_f1"])
                    # results = {'micro_f1': np.mean(mif1s), 'macro_f1': np.mean(maf1s), 'accuracy': np.mean(accuracies)}
                    test_score = np.mean(accuracies)
                    t_2 = time.time()
                    print("Epoch: {:03d}, acc: {:.2f}, time consumption: {:.2f}s".format(epoch, test_score, t_2 - t_1))
                # print("Epoch: {:03d}, acc: {:.2f}, time consumption: {:.2f}s".format(epoch, test_score, t_2 - t_1))
                if best_test < test_score:
                    best_test = test_score
                trial.report(test_score, epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
        return best_test

    return objective



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCL Arguments.')
    parser.add_argument('--dataset', type=str, default="HIV")
    parser.add_argument('--model', type=str, choices=["GraphCL"], default="GraphCL")
    parser.add_argument('--encoder', type=str, choices=["GConv"], default="GConv")
    parser.add_argument("--device", type=int, default=5)
    args = parser.parse_args()
    args = arg_parse(parser, args.model, args.encoder, args.dataset)
    if args.search == True:
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
    else:
        pass
