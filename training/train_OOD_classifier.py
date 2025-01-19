import time

import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import models.transform_layers as TL
from utils.utils import AverageMeter
import os
import numpy as np
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)

def train(P, epoch, model, criterion, criterion_supcon, criterion_comp, optimizer,
          scheduler, loader, p_mask, mask_back, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    joint_linear = model.joint_distribution_layer

    if epoch == 1:
        # define optimizer and save in P (argument)
        milestones = [int(0.6 * P.epochs), int(0.75 * P.epochs), int(0.9 * P.epochs)]

        joint_linear_optim = torch.optim.SGD(joint_linear.parameters(),
                                             lr=1e-1, weight_decay=P.weight_decay)
        P.joint_linear_optim = joint_linear_optim
        P.joint_scheduler = lr_scheduler.MultiStepLR(P.joint_linear_optim, gamma=0.1, milestones=milestones)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['loss'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        labels = labels % P.n_cls_per_task
        model.eval()
        count = n * P.n_gpus

        data_time.update(time.time() - check)
        check = time.time()

        # horizontal flip
        batch_size = images.size(0)
        images = images.to(device)
        images = hflip(images)

        # rotation B -> 4B
        labels = labels.to(device)
        images = torch.cat([torch.rot90(images, rot, (2, 3)) for rot in range(4)])

        # Assign each rotation degree a different class
        joint_labels = torch.cat([labels + P.n_cls_per_task * i for i in range(4)], dim=0)

        images = simclr_aug(images)

        # Obtain features from fixed feature extractor
        with torch.no_grad():
            _, outputs_aux, masks = model(P.t, images, s=P.smax, penultimate=True, feature=True)
        penultimate = outputs_aux['penultimate'].detach()

        # obtain outputs
        outputs_joint = joint_linear[P.t](penultimate)

        joint_labels = joint_labels.type(torch.long)
        loss_joint = criterion(outputs_joint, joint_labels)

        P.joint_linear_optim.zero_grad()

        loss_joint.backward()

        P.joint_linear_optim.step()

        lr = P.joint_linear_optim.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        losses['loss'].update(loss_joint.item(), batch_size)

        if count % 50 == 0:
            P.logger.print('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                           '[Loss %f]' %
                           (epoch, count, batch_time.value, data_time.value, lr,
                            losses['loss'].value))
        check = time.time()

    P.joint_scheduler.step()

    P.logger.print('[DONE] [Time %.3f] [Data %.3f] [Loss %f]' %
                   (batch_time.average, data_time.average,
                    losses['loss'].average))