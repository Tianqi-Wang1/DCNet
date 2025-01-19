import torch
from utils.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cil(P, model, loaders):

    mode = model.training
    model.eval()

    for data_id, loader in loaders.items():
        error_top1 = AverageMeter()
        outputs_all, targets_all = [], []
        scores_all = []
        for n, (images, labels) in enumerate(loader): # loader is in_loader
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            # For a batch, obtain outputs from each task networks and concatenate for cil
            cil_outputs = torch.tensor([]).to(device)
            scores_batch = []
            for t in range(P.t + 1):
                # For ensemble prediction
                outputs = 0
                for i in range(4):

                    rot_images = torch.rot90(images, i, (2, 3))
                    with torch.no_grad():
                        _, outputs_aux, _ = model(t, rot_images, s=P.smax, joint=True, penultimate=True)

                    out = outputs_aux['joint']

                    output_ = out[:, P.n_cls_per_task * i: P.n_cls_per_task * (i + 1)] / 4.

                    outputs += output_

                # Apply calibration if available
                new_outputs = outputs
                # new_outputs = outputs * confidence
                cil_outputs = torch.cat((cil_outputs, new_outputs.detach()), dim=1)

                scores, _ = torch.max(new_outputs, dim=1, keepdim=True)
                scores_batch.append(scores)

            # Top 1 error. Accuracy is 100 - error.
            top1, = error_k(cil_outputs.data, labels, ks=(1,))
            error_top1.update(top1.item(), batch_size)

            outputs_all.append(cil_outputs.data.cpu().numpy())
            targets_all.append(labels.data.cpu().numpy())

            scores_batch = torch.cat(scores_batch, dim=1)
            scores_all.append(scores_batch.cpu().numpy())

            if n % 100 == 0:
                P.logger.print('[Test %3d] [Test@1 %.3f]' %
                     (n, 100-error_top1.value))

        P.logger.print('[Data id %3d] [ACC@1 %.3f]' %
             (data_id, 100-error_top1.average))
        if P.mode == 'cil':
            P.cil_tracker.update(100 - error_top1.average, P.t, data_id)
        else:
            raise NotImplementedError()

    model.train(mode)
    return error_top1.average

def error_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results

def test_classifier(P, model, loaders, steps, marginal=False, logger=None, Test=True):

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for data_id, loader in loaders.items():
        error_top1 = AverageMeter()
        error_calibration = AverageMeter()
        with torch.no_grad():
            for n, (images, labels) in enumerate(loader):
                labels = labels % P.n_cls_per_task
                batch_size = images.size(0)

                images, labels = images.to(device), labels.to(device)

                if marginal:
                    # Ensemble
                    outputs = 0
                    for i in range(4):
                        rot_images = torch.rot90(images, i, (2, 3))
                        _, outputs_aux, _ = model(data_id, rot_images, s=P.smax, joint=True, penultimate=True)
                        outputs += outputs_aux['joint'][:, P.n_cls_per_task * i: P.n_cls_per_task * (i + 1)] / 4.
                else:
                    outputs, _ = model(data_id, images, s=P.smax)

                # Top 1 error. Acc is 100 - error
                top1, = error_k(outputs.data, labels, ks=(1,))
                error_top1.update(top1.item(), batch_size)

                # Accuracy at 100th batch just for reference
                if n % 100 == 0:
                    P.logger.print('[Test %3d] [Test@1 %.3f]' %
                        (n, 100-error_top1.value))

        # Accuracy over entire batch (i.e. the final accuracy)
        P.logger.print(' * [ACC@1 %.3f]' %
             (100-error_top1.average))

        # Record the test accuracy
        if marginal and Test:
            P.til_tracker.update(100 - error_top1.average,
                                int(P.logout.split('task_')[-1]),
                                data_id)

    model.train(mode)

    return error_top1.average
