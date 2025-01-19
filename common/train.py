import os
import torch
import torch.nn as nn
import torch.optim as optim
from .sgd_hat import SGD_hat
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from common.common import parse_args
import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
from utils.utils import load_checkpoint, Logger
from utils.util_loss import (CompLoss, SupConLoss)

# torch.cuda.set_device(0)
P = parse_args()

P.logger = Logger(P)

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = 1

### only use one ood_layer while training
P.ood_layer = P.ood_layer[0]

### Initialize dataset ###
train_set_, _, image_size, n_cls_per_task, total_cls = get_dataset(P, dataset=P.dataset, download=True)
P.image_size = image_size
P.n_cls_per_task = n_cls_per_task
P.total_cls = total_cls
P.n_tasks = int(total_cls // n_cls_per_task)

cls_list = get_superclass_list(P.dataset)

train_set = get_subclass_dataset(P, train_set_, classes=cls_list[P.t], f_select=0.9)
test_set = get_subclass_dataset(P, train_set_, classes=cls_list[P.t], l_select=0.9)

kwargs = {'pin_memory': False, 'num_workers': 0}

train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
test_loaders = {P.t: DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)}

### Initialize model ###
simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
P.shift_trans = P.shift_trans.to(device)

model = C.get_classifier(P, P.model, n_classes=P.n_cls_per_task).to(device)
model = C.get_shift_classifer(P.n_tasks, model, P.K_shift).to(device)

criterion = nn.CrossEntropyLoss().to(device)
criterion_supcon = SupConLoss(temperature=P.temp).to(device)

if P.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
elif P.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=P.lr_init, betas=(.9, .999), weight_decay=P.weight_decay)
    lr_decay_gamma = 0.3
elif P.optimizer == 'lars':
    # from torchlars import LARS
    from .lars_optimizer import LARC
    base_optimizer = SGD_hat(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    optimizer = LARC(base_optimizer, eps=1e-8, trust_coefficient=0.001)
    lr_decay_gamma = 0.1
else:
    raise NotImplementedError()

if P.lr_scheduler == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
elif P.lr_scheduler == 'step_decay':
    milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
else:
    raise NotImplementedError()

from training.scheduler import GradualWarmupScheduler
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)

# if resume_path is for resuming training
if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state, strict=not P.no_strict)
    optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch']
    best = 100
    error = 100.0
else:
    resume = False
    start_epoch = 1
    best = 100.0
    error = 100.0

if P.mode == 'train_OOD_classifier' or P.t > 0:
    assert P.load_path is not None

    checkpoint = torch.load(os.path.join(P.load_path, 'last.model'))
    model.load_state_dict(checkpoint, strict=not P.no_strict)

# Load masks.
if P.load_path is None:
    p_mask = None
    mask_back = None
else:
    P.logger.print("Loading previous masks")

    mask_checkpoints = torch.load(os.path.join(P.load_path, 'masks'))
    p_mask = mask_checkpoints['p_mask']
    mask_back = mask_checkpoints['mask_back']

    for n, p in model.named_parameters():
        p.grad = None
        if n in mask_back.keys():
            p.hat = mask_back[n]
        else:
            p.hat = None
    
    reg = 0
    count = 0
    for m in p_mask.values():
        reg += m.sum()
        count += np.prod(m.size()).item()
    reg /= count
    P.temp = max(P.temp*(1 - reg), 0.05)
criterion_comp = CompLoss(P, temperature=P.temp).to(device)
