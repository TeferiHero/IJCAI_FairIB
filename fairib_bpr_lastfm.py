
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import datetime
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from models.fairib_bpr_item import FairIB_BPR_Item
from utils import *
from models import *
from tqdm import tqdm
from utils.general import *
import scipy.stats as stats

import pdb
import sys

if __name__ == '__main__':

    args = parse_input_args('lastfm_bpr_fairib',backbone='gcn', dataset='lastfm-360k',
                            n_layers=0, batch_size=4096, log_path='logs/ib_bpr_fm_',
                            param_path='param/ib_bpr_item_fm_', pretrain_path=None,
                            num_epochs=1000, device='cpu', beta=2, gamma=1, sigma=0.4)

    init_from_arguments(args)

    (train_u2i, train_i2u,
     test_u2i, test_i2u,
     train_set, test_set, user_side_features,
     n_users, n_items) = load_dataset(args.dataset)

    u_sens = user_side_features['gender'].astype(np.int32)

    if args.pretrain_path is not None and args.pretrain_path != '':
        evaluate_pretrained(n_users, n_items, train_u2i, test_u2i, u_sens, args)
        exit()



    dataset = BPRTrainLoader(train_set, train_u2i, n_items)

    graph = Graph(n_users, n_items, train_u2i)
    norm_adj = graph.generate_ori_norm_adj()

    sens_enc = SemiGCN(n_users, n_items, norm_adj,
                       args.emb_size, 3, args.device,
                       nb_classes=np.unique(u_sens).shape[0])
    train_semigcn(sens_enc, u_sens, n_users, device=args.device)

    
    fair_ib = FairIB_BPR_Item(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)
    fair_ib.to(args.device)
    train_unify_mi('bpr', sens_enc, fair_ib, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args)
    sys.stdout = None
