import os
import pickle
import random

import sys
from datetime import datetime
from torchvision import datasets, transforms
import numpy as np
import torch
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from PIL import Image
import faiss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value # THIS IS THE CURRENT VALUE
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def load_checkpoint(logdir, mode='last', loc=None):
    if mode == 'last':
        model_path = os.path.join(logdir, 'last.model')
        optim_path = os.path.join(logdir, 'last.optim')
        config_path = os.path.join(logdir, 'last.config')
    elif mode == 'best':
        model_path = os.path.join(logdir, 'best.model')
        optim_path = os.path.join(logdir, 'best.optim')
        config_path = os.path.join(logdir, 'best.config')

    else:
        raise NotImplementedError()

    print("=> Loading checkpoint from '{}'".format(logdir))
    if os.path.exists(model_path):
        model_state = torch.load(model_path, map_location=loc)
        optim_state = torch.load(optim_path, map_location=loc)
        with open(config_path, 'rb') as handle:
            cfg = pickle.load(handle)
    else:
        return None, None, None

    return model_state, optim_state, cfg


def save_checkpoint(epoch, model_state, optim_state, logdir):
    last_model = os.path.join(logdir, 'last.model')
    last_optim = os.path.join(logdir, 'last.optim')
    last_config = os.path.join(logdir, 'last.config')

    opt = {
        'epoch': epoch,
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_linear_checkpoint(linear_optim_state, logdir):
    last_linear_optim = os.path.join(logdir, 'last.linear_optim')
    torch.save(linear_optim_state, last_linear_optim)

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

class Logger:
    def __init__(self, P):
        self.init = datetime.now()
        self.local_rank = P.local_rank
        self.P = P

        if P.load_path is None:
            if P.mode == 'train_IOE_DAC' and P.t == 0:
                pass
            elif P.mode == 'train_IOE_DAC' and P.t > 0:
                self.P.load_path = f'./logs/{P.dataset}/linear_task_{P.t - 1}'
            elif P.mode == 'train_OOD_classifier':
                self.P.load_path = f'./logs/{P.dataset}/feature_task_{P.t}'
            elif P.mode == 'cil':
                self.P.load_path = f'./logs/{P.dataset}/linear_task_{P.t}'
            else:
                raise NotImplementedError()

        if P.logout is None:
            if P.mode == 'train_IOE_DAC' and P.t == 0:
                self.P.logout = f'./logs/{P.dataset}/feature_task_{P.t}'
            elif P.mode == 'train_IOE_DAC' and P.t > 0:
                self.P.logout = f'./logs/{P.dataset}/feature_task_{P.t}'
            elif P.mode == 'train_OOD_classifier':
                self.P.logout = f'./logs/{P.dataset}/linear_task_{P.t}'
            elif P.mode == 'cil':
                self.P.logout = f'./logs/{P.dataset}/linear_task_{P.t}'
            else:
                raise NotImplementedError()

        self._make_dir()

    def _make_dir(self):
        if self.local_rank == 0:
            if not os.path.isdir(self.dir()):
                os.makedirs(self.dir())

    def dir(self):
        return self.P.logout

    def print(self, *object, sep=' ', end='\n', flush=False, filename='/result.txt'):
        if self.local_rank == 0:
            print(*object, sep=sep, end=end, file=sys.stdout, flush=flush)

            if self.P.printfn is not None:
                filename = self.P.printfn
            with open(self.dir() + '/' + filename, 'a') as f:
                print(*object, sep=sep, end=end, file=f, flush=flush)

class Tracker:
    def __init__(self, P):
        self.print = P.logger.print
        self.mat = np.zeros((P.n_tasks * 2 + 1, P.n_tasks * 2 + 1)) - 100

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(self.mat[task_id, :p_task_id + 1])

        # Compute forgetting
        for i in range(task_id):
            self.mat[-1, i] = self.mat[i, i] - self.mat[task_id, i]

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='acc', print=None):
        if print is None: print = self.print
        if type == 'acc':
            # Print accuracy
            for i in range(task_id + 1):
                for j in range(task_id + 1):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                print("{:.2f}".format(self.mat[i, -1]))
        elif type == 'forget':
            # Print forgetting and average incremental accuracy
            for i in range(task_id + 1):
                acc = self.mat[-1, i]
                if acc != -100:
                    print("{:.2f}\t".format(acc), end='')
                else:
                    print("\t", end='')
            print("{:.2f}".format(self.mat[-1, -1]))
            if task_id > 0:
                forget = np.mean(self.mat[-1, :task_id])
                print("{:.2f}".format(forget))
        else:
            raise NotImplementedError("Type must be either 'acc' or 'forget'")