from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of DCNet')

    parser.add_argument('--dataset', help='Dataset',default='cifar100_10t',
                        choices=['cifar100_10t', 'cifar100_20t',
                        'tinyImagenet_10t', 'tinyImagenet_20t', 'Imagenet100_10t', 'Imagenet100_20t'], type=str)
    parser.add_argument('--model', help='Model',
                        default='resnet18', choices=['resnet18'], type=str)
    parser.add_argument('--mode', help='Training/inference mode',
                        default='train_IOE_DAC', type=str, choices=['train_IOE_DAC', 'train_OOD_classifier',
                                                            'ood', 'test_marginalized_acc',
                                                            'cil', 'cil_pre'])
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=512, type=int)

    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='rotation',
                        choices=['rotation', 'none'], type=str)

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint, masks, AND learned calibration',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=1, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save models',
                        default=1, type=int)

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=2, type=int)
    parser.add_argument('--DAC_epoch', help='Epoch for DAC',
                        default=400, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'adam', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay (HAT tends to break with weight_decay)',
                        default=0.0, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=64, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=100, type=int)
    parser.add_argument('--printfn', help='log file name. Must end with .txt',
                        default='results.txt', type=str)

    ##### Evaluation Configurations #####
    parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                        default=None, nargs="*", type=str)
    parser.add_argument("--ood_score", help='score function for OOD detection',
                        default=['baseline_marginalized'], nargs="+", type=str)
    parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr', 'shift'],
                        default=['simclr', 'shift'], nargs="+", type=str)
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                        action='store_true')

    parser.add_argument("--print_score", help='print quantiles of ood score',
                        action='store_true')
    parser.add_argument("--save_score", help='save ood score for plotting histogram',
                        action='store_true')

    parser.add_argument('--data_path', help='data path',
                        default='./data', type=str)
    parser.add_argument('--t', help='Task id, should change over time. Use for loading dataset of task t',
                        default=0, type=int)
    parser.add_argument('--smax', help='Max value for s',
                        default=700, type=float)
    parser.add_argument('--logout', help='Log directory',
                        default=None, type=str)
    parser.add_argument('--lamb0', help='Hyper-param for HAT regularzation at task 0',
                        type=float, default=1)
    parser.add_argument('--lamb1', help='Hyper-param for HAT regularzation at task > 0',
                        type=float, default=0.75)
    parser.add_argument('--all_dataset', help='use all datasets of learned tasks',
                        action='store_true')
    parser.add_argument('--amp', help='use automatic mixed prevision',
                        action='store_true')

    parser.add_argument('--w', default=1, type=float,
                        help='loss scale')
    parser.add_argument('--feat_dim', default = 256, type=int,
                        help='feature dim')
    parser.add_argument('--embedding_dim', default = 512, type=int,
                        help='embedding_dim dim')
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')
    parser.add_argument('--temp_sim', type=float, default=0.1, help='temperature for loss function')

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()
