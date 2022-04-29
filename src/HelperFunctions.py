import os
import random
from datetime import datetime

from scipy.stats import spearmanr, pearsonr
import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_now():
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S%f")

def get_device(device):
    if type(device) == torch.device:
        return device

    if device == 'cpu':
        return torch.device('cpu')
    else:
        return torch.device(0)

def get_metrics(sys_scores, gold_scores, tags):
    rets = {}
    pearsonr_scores, spearmanr_scores = {}, {}
    mean_pearsonr_scores, mean_spearmanr_scores = [], []
    for tag in set(tags):
        ss, gs = [], []
        for s, g, t in zip(sys_scores, gold_scores, tags):
            if t == tag:
                ss.append(s)
                gs.append(g)
        rets[f'pearsonr-{tag}'] = pearsonr(ss, gs)[0]
        rets[f'spearmanr-{tag}'] = spearmanr(ss, gs)[0]
        mean_pearsonr_scores.append(rets[f'pearsonr-{tag}'])
        mean_spearmanr_scores.append(rets[f'spearmanr-{tag}'])
    rets[f'pearsonr-all'] = pearsonr(sys_scores, gold_scores)[0]
    rets[f'spearmanr-all'] = spearmanr(sys_scores, gold_scores)[0]
    rets[f'pearsonr-wmean'] = sum(mean_pearsonr_scores) / len(mean_pearsonr_scores)
    rets[f'spearmanr-wmean'] = sum(mean_spearmanr_scores) / len(mean_spearmanr_scores)

    return rets

