import time
import numpy as np
import torch.optim
import torch.nn.functional as F
import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, Supervised_NT_xent
from utils.utils import AverageMeter, normalize
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)

def update_s(P, b, B):
    """ b: current batch, B: total num batch """
    s = (P.smax - 1 / P.smax) * b / B + 1 / P.smax
    return s

def compensation(P, model, thres_cosh=50, s=1):
    """ Equation before Eq. (4) in HAT paper """
    for n, p in model.named_parameters():
        if 'ec' in n:
            if p.grad is not None:
                num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad *= P.smax / s * num / den

def compensation_clamp(model, thres_emb=6):
    # Constrain embeddings
    for n, p in model.named_parameters():
        if 'ec' in n:
            if p.grad is not None:
                p.data.copy_(torch.clamp(p.data, -thres_emb, thres_emb))

def hat_reg(P, p_mask, masks):
    """ masks and p_mask must have values in the same order """
    reg, count = 0., 0.
    if p_mask is not None:
        for m, mp in zip(masks, p_mask.values()):
            aux = 1. - mp#.to(device)
            reg += (m * aux).sum()
            count += aux.sum()
        reg /= count
        return P.lamb1 * reg
    else:
        for m in masks:
            reg += m.sum()
            count += np.prod(m.size()).item()
        reg /= count
        return P.lamb0 * reg

def train(P, epoch, model, criterion, criterion_supcon, criterion_comp, optimizer, scheduler, loader, p_mask, mask_back, logger=None,
          simclr_aug=None, linear=None, linear_optim=None, thres_cosh=50, thres_emb=6):

    if epoch == P.DAC_epoch:
        model.eval()
        feature_list = None
        label_list = None
        for n, (images, labels) in enumerate(loader):
            # labels = labels
            labels = labels % P.n_cls_per_task
            model.eval()

            images = images.to(device)

            images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)

            # Data rotation 2B -> 8B
            images1 = torch.cat([torch.rot90(images, rot, (2, 3)) for rot in range(4)])
            images2 = torch.cat([torch.rot90(images2, rot, (2, 3)) for rot in range(4)])
            images_pair = torch.cat([images1, images2], dim=0)
            images_pair = simclr_aug(images_pair)

            # Assign each rotation degree a different class
            joint_labels = torch.cat([labels + P.n_cls_per_task * i for i in range(4)], dim=0)
            joint_labels = joint_labels.repeat(2)

            with torch.no_grad():
                _, outputs_aux, masks = model(P.t, images_pair, s=P.smax, penultimate=True, feature=True)
            feature_original = outputs_aux['feature'].detach()
            feature = F.normalize(feature_original, dim=1)

            if n == 0:
                feature_list = feature.cpu().numpy()
                label_list = joint_labels.cpu().numpy()
            else:
                feature_list = np.concatenate((feature_list, feature.cpu().numpy()))
                label_list = np.concatenate((label_list, joint_labels.cpu().numpy()))


        if not os.path.exists(P.logout + '/md'):
            os.makedirs(P.logout + '/md')

        level_list = []
        for y in range(P.n_cls_per_task * 4):
            idx = np.where(label_list == y)[0]
            f = feature_list[idx]

            feat_dot_prototype = np.matmul(f, criterion_comp.prototypes[P.n_cls_per_task * P.t * 4:P.n_cls_per_task * (P.t + 1) * 4, :][y: y+1,:].cpu().numpy().T)
            feat_dot_prototype = np.mean(feat_dot_prototype)

            level_list.append(abs(feat_dot_prototype))
        mean = np.mean(np.array(level_list))
        np.save(P.logout + '/md' + f'/aggregation_mean_{P.t}', mean)

        if P.t > 0:
            P.temp_sim = max(P.temp_sim * mean / P.mean_all, 0.05)

    enabled = False
    if P.amp:
        enabled = True
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    assert simclr_aug is not None

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['sim'] = AverageMeter()
    losses['comp'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        labels = labels % P.n_cls_per_task
        model.train()
        count = n * P.n_gpus

        data_time.update(time.time() - check)
        check = time.time()

        # Update degree of step
        s = update_s(P, n, len(loader))

        # Data augmentation B -> 2B
        batch_size = images.size(0)
        images = images.to(device)
        images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)

        # Data rotation 2B -> 8B
        images1 = torch.cat([torch.rot90(images1, rot, (2, 3)) for rot in range(4)])
        images2 = torch.cat([torch.rot90(images2, rot, (2, 3)) for rot in range(4)])
        images_pair = torch.cat([images1, images2], dim=0)

        # Assign each rotation a class
        labels = labels.to(device)
        rot_sim_labels = torch.cat([labels + P.n_cls_per_task * i for i in range(4)], dim=0)
        rot_sim_labels = rot_sim_labels.repeat(2).to(device)

        images_pair = simclr_aug(images_pair)

        with torch.cuda.amp.autocast(enabled=enabled):
            _, outputs_aux, masks = model(P.t, images_pair, s=s, feature=True, penultimate=True)

            feature = normalize(outputs_aux['feature'])

            if epoch > P.DAC_epoch:
                sim_matrix = get_similarity_matrix(feature[0:rot_sim_labels.size(0)])
                loss_sim = Supervised_NT_xent(sim_matrix, labels=rot_sim_labels,
                                          temperature=P.temp_sim)
                comp_loss = criterion_comp(feature, criterion_comp.prototypes[P.n_cls_per_task*P.t*4:P.n_cls_per_task*(P.t+1)*4,:], rot_sim_labels)
                loss = comp_loss + P.w * loss_sim
            else:
                comp_loss = criterion_comp(feature, criterion_comp.prototypes[P.n_cls_per_task*P.t*4:P.n_cls_per_task*(P.t+1)*4,:], rot_sim_labels)
                loss_sim = comp_loss
                loss = comp_loss
            # HAT regularization
            loss += hat_reg(P, p_mask, masks)

        optimizer.zero_grad()

        hat = False
        if P.t > 0:
            hat = True

        if P.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # embedding gradient compensation. Refer to HAT
        compensation(P, model, thres_cosh, s=s)

        if P.amp:
            if P.optimizer == 'lars':
                scaler.step(optimizer, hat=hat)
                scaler.update()
            else:
                raise NotImplementedError("feature training must be protected by HAT")
        else:
            if P.optimizer == 'lars':
                optimizer.step(hat=hat)
            else:
                raise NotImplementedError("feature training must be protected by HAT")

        # clamp embedding
        compensation_clamp(model, thres_emb)

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        # Train the standard classifier without ensemble for reference.
        penul_1 = outputs_aux['penultimate'][:batch_size]
        penul_2 = outputs_aux['penultimate'][4 * batch_size:5 * batch_size]
        outputs_aux['penultimate'] = torch.cat([penul_1, penul_2])

        outputs_linear_eval = linear[P.t](outputs_aux['penultimate'].detach())

        loss_linear = criterion(outputs_linear_eval, labels.repeat(2).type(torch.long))

        linear_optim.zero_grad()
        loss_linear.backward()
        linear_optim.step()

        losses['sim'].update(loss_sim.item(), batch_size)
        losses['comp'].update(comp_loss.item(), batch_size)

        if count % 50 == 0:
            P.logger.print('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[Losscomp %f] [Losssim %f]'%
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['comp'].value, losses['sim'].value))

        check = time.time()

    P.logger.print('[DONE] [Time %.3f] [Data %.3f] [Losscomp %f] [Losssim %f]' %
         (batch_time.average, data_time.average, losses['comp'].average, losses['sim'].average))
