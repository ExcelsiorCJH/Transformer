import os
import glob
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist


def fix_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_trained_model(config):
    version = config.version
    dataset_name = config.dm.dataset_name
    task_name = config.model.task_name

    ckpt_list = glob.glob(
        os.path.join(
            config.train.ckpt_dir,
            f"version-{version}-{task_name}-{dataset_name.lower()}/*.pt",
        )
    )
    ckpt_list = sorted(ckpt_list, key=lambda e: e.split("_")[-1], reverse=True)

    state_dict = torch.load(ckpt_list[-1])
    return state_dict


def plot(true, preds=None, fname="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(fname, bbox_inches="tight")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, logging):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
