import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import parse_args
import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
from utils.utils import Logger, Tracker

P = parse_args()

P.logger = Logger(P)

# P.n_gpus = torch.cuda.device_count()
P.n_gpus = 1

P.multi_gpu = False

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

train_set, test_set, image_size, n_cls_per_task, total_cls = get_dataset(P, dataset=P.dataset, download=True)
cls_list = get_superclass_list(P.dataset)
P.n_superclasses = len(cls_list)

kwargs = {'pin_memory': False, 'num_workers': 0}
P.image_size = image_size
P.n_cls_per_task = n_cls_per_task
P.total_cls = total_cls
P.n_tasks = int(total_cls // n_cls_per_task)


if P.mode == 'cil':
    test_loaders = {}
    if P.all_dataset:
        # report the accuracies of all the tasks learned so far
        last_learned_task = int(P.load_path.split('task_')[-1])

        for p_task_id in range(last_learned_task + 1):
            # Obtain test data
            test_subset = get_subclass_dataset(P, test_set, classes=cls_list[p_task_id])
            test_loaders[p_task_id] = DataLoader(test_subset, shuffle=False, batch_size=P.test_batch_size, **kwargs)

simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
P.shift_trans = P.shift_trans.to(device)

model = C.get_classifier(P, P.model, n_classes=P.n_cls_per_task).to(device)
model = C.get_shift_classifer(P.n_tasks, model, P.K_shift).to(device)
criterion = nn.CrossEntropyLoss().to(device)

# Load model
checkpoint = torch.load(os.path.join(P.load_path, 'last.model'))
model.load_state_dict(checkpoint, strict=not P.no_strict)
