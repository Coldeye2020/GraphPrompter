import torch
from torch_geometric.data import DataLoader
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
from code.GraphPrompter.model.model import DGCL
import GCL.losses as L
from GCL.models import DualBranchContrast
import optuna
import time
from utils.dataset import TUDataset_aug, OGBDataset_aug
from utils.utils import setup_seed, evaluate_embedding, arg_parse
from utils.loader import get_split_loader
from GCL.augmentors import *
import argparse

import warnings
warnings.filterwarnings('ignore')



class MainTrainer():
    def __init__(self, cf):
        self.__dict__.update(cf.__dict__)
        self.loader, self.train_loader, self.val_loader, self.test_loader = get_split_loader(name=cf.dataset, root=cf.root, train_ratio=cf.train_ratio, val_ratio=cf.val_ratio, batch_size=cf.batch_size, num_workers=cf.num_workers)
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
        if cf.encoder == "DGCL":
            # model = DGCL()
            pass
        elif cf.encoder == "GIN":
            # model = GIN()
            pass
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=cf.tau), mode='G2G', intraview_negs=cf.intra_negative).to(cf.device)
        # self.encoder_model = Encoder(encoder=model, augmentor=(aug1, aug2), contrast_model=self.contrast_model, cf=cf).to(cf.device)

        # if cf.use_scheduler:
        #     self.optimizer = torch.optim.Adam(self.encoder_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
        #     self.scheduler = CosineDecayScheduler(max_val=cf.lr, warmup_steps=cf.epochs//10, total_steps=cf.epochs)
        # else:
        self.optimizer = torch.optim.Adam(self.encoder_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)

