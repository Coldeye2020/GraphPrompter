import torch
from torch_geometric.data import DataLoader
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
from code.GraphPrompter.model.model import DGCL
import optuna
import time
from utils.dataset import TUDataset_aug, OGBDataset_aug
from utils.utils import setup_seed, evaluate_embedding, arg_parse
from utils.loader import get_split_loader
from GCL.augmentors import *
from GCL.models import DualBranchContrast

import warnings
warnings.filterwarnings('ignore')



class BaseTrainer():
    def __init__(self, cf):
        self.__dict__.update(cf.__dict__)
        self.cf = cf
        # self.evaluator = Evaluator(cf.dataset)

        if cf.two_aug:
            aug1 = [Identity(),
                    RWSampling(num_seeds=1000, walk_length=10),
                    NodeDropping(pn=0.1),
                    FeatureMasking(pf=0.1),
                    EdgeRemoving(pe=0.1)]
            aug2 = [Identity(),
                    RWSampling(num_seeds=1000, walk_length=10),
                    NodeDropping(pn=0.1),
                    FeatureMasking(pf=0.1),
                    EdgeRemoving(pe=0.1)]

        else:
            aug1 = Identity()
            aug2 = [RWSampling(num_seeds=1000, walk_length=10),
                    NodeDropping(pn=0.1),
                    FeatureMasking(pf=0.1),
                    EdgeRemoving(pe=0.1)]
        if cf.encoder == "DGCL"
            # model = GConv(input_dim=cf.feat_dim, hidden_dim=cf.n_hidden, num_layers=cf.n_layer, dropout= cf.dropout).to(cf.device)
            pass
        elif cf.encoder == "GraphPrompter":
            # model = GPropConv(input_dim=cf.feat_dim, hidden_dim=cf.n_hidden, num_layers=cf.n_layer, dropout= cf.dropout).to(cf.device)
            pass
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=cf.tau), mode='G2G', intraview_negs=cf.intra_negative).to(cf.device)
        self.encoder_model = Encoder(encoder=model, augmentor=(aug1, aug2), contrast_model=self.contrast_model, cf=cf).to(cf.device)

        if cf.use_scheduler:
            self.optimizer = th.optim.Adam(self.encoder_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
            self.scheduler = CosineDecayScheduler(max_val=cf.lr, warmup_steps=cf.epochs//10, total_steps=cf.epochs)
        else:
            self.optimizer = th.optim.Adam(self.encoder_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)

def objective(trial):
    study = optuna.create_study(direction="maximize")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Training settings")
    # parser = ACDGCLConfig.add_exp_setting_args(parser)
    # exp_args = parser.parse_known_args()[0]
    # parser = ACDGCLConfig(exp_args).add_model_specific_args(parser)
    # args = parser.parse_args()
    # 超参数

    # 数据

    # 模型

    # 优化器
    