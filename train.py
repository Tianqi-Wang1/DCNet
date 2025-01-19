CUDA_VISIBLE_DEVICES=0
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint

from common.train import *
from evals import test_classifier

P.logger.print(P)

if P.model == 'resnet18':
    from mask_ops import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if P.mode == 'train_OOD_classifier':
    from training.train_OOD_classifier import train

elif P.mode == 'train_IOE_DAC':
    from training.train_IOE_DAC import train

else:
    raise NotImplementedError()

linear = model.linear

linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

mean_all = 0
if os.path.exists(f'./logs/{P.dataset}/feature_task_{P.t-1}' + '/md' + f'/aggregation_mean_{P.t-1}.npy'):
    P.logger.print("*** Load Statistics for Aggregation ***")
    for i in range (P.t):
        level_temp = np.load(f'./logs/{P.dataset}/feature_task_{i}' + '/md' + f'/aggregation_mean_{i}.npy')
        mean_all += level_temp
    P.mean_all = mean_all/P.t

# Training
try:
    for epoch in range(start_epoch, P.epochs + 1):
        P.logger.print(f'Epoch {epoch}', P.logout, f'Tempature {P.temp_sim}')
        model.train()

        kwargs = {}
        kwargs['linear'] = linear
        kwargs['linear_optim'] = linear_optim
        kwargs['simclr_aug'] = simclr_aug

        train(P, epoch, model, criterion, criterion_supcon, criterion_comp, optimizer, scheduler_warmup, train_loader, p_mask, mask_back, **kwargs)

        model.eval()

        if epoch % P.save_step == 0 and P.local_rank == 0:
            save_states = model.state_dict()

            save_checkpoint(epoch, save_states, optimizer.state_dict(), P.logout)
            save_linear_checkpoint(linear_optim.state_dict(), P.logout)

        if epoch % P.error_step == 0 and ('train' in P.mode):
            if P.mode == 'train_OOD_classifier':
                error = test_classifier(P, model, test_loaders, epoch, marginal=True, Test=False)
            elif P.mode == 'train_IOE_DAC':
                error = test_classifier(P, model, test_loaders, epoch, marginal=False)

            is_best = (best > error)
            if is_best:
                best = error

            P.logger.print('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))

except KeyboardInterrupt:
    P.logger.print()

# Update and save masks
if P.local_rank == 0:
    checkpoint = torch.load(P.logout + '/last.model', map_location=None)

    model.load_state_dict(checkpoint)

    p_mask = cum_mask(P, P.t, model, p_mask)
    mask_back = freeze_mask(P, P.t, model, p_mask)
    p_mask = cum_mask_simclr(P, P.t, model, p_mask)
    mask_back = freeze_mask_simclr(P, P.t, model, p_mask, mask_back)
    checkponts = {'p_mask': p_mask,
                  'mask_back': mask_back}
    torch.save(checkponts, P.logout + '/masks')
    P.logger.print("Saved masks")
P.logger.print('\n\n\n\n\n\n\n\n')
