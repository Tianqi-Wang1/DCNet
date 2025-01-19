from common.eval import *

model.eval()

P.logger.print(P)
if P.dataset == 'cifar100_10t':
    P.n_cls_per_task = 10
elif P.dataset == 'cifar100_20t':
    P.n_cls_per_task = 5


P.cil_tracker = Tracker(P)
P.til_tracker = Tracker(P)
if os.path.exists(f'./logs/{P.dataset}/cil_acc_tracker'):
    P.cil_tracker.mat = torch.load(f'./logs/{P.dataset}/cil_acc_tracker')
if os.path.exists(f'./logs/{P.dataset}/til_acc_tracker'):
    P.til_tracker.mat = torch.load(f'./logs/{P.dataset}/til_acc_tracker')

# CIL inference
if P.mode == 'cil':
    from evals import cil

    with torch.no_grad():
        cil(P, model, test_loaders)

else:
    raise NotImplementedError()

P.logger.print()

if P.mode == 'cil':
    P.logger.print("CIL result")
    P.cil_tracker.print_result(P.t, type='acc')
    P.cil_tracker.print_result(P.t, type='forget')

torch.save(P.cil_tracker.mat, f'./logs/{P.dataset}/cil_acc_tracker')

P.logger.print('\n\n\n\n\n\n\n\n')
