import argparse
import logging
import os
import random
import sys
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn.parallel
import torch.optim
import torch.optim
import torch.utils.data
import torch.utils.data
import torchvision

from attacks import PGD, EPGD
from logger import CsvLogger
from models.wideresnet import wideresnet28, wideresnet34
from sample_methods.clean import Clean
from sample_methods.gaussian import Gaussian


def get_args():
    parser = argparse.ArgumentParser(description='PNI training with PyTorch')
    parser.add_argument('--data', default='./data', metavar='PATH', help='Path to data')
    parser.add_argument('--dataset', default='cifar10', metavar='SET', help='Dataset (CIFAR-10, CIFAR-100, ImageNet)')

    parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
    parser.add_argument('--data_parallel', action='store_true', help='Only supported for wideresnet')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')
    parser.add_argument('--print-model', action='store_true', default=False, help='print model to stdout')

    # Optimization options
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--val-batch-size', default=256, type=int, metavar='N',
                        help='validation mini-batch size (default: 256)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The learning rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=1e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 200, 300, 400],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--opt', type=str, default='sgd', help='Optimizer')

    parser.add_argument('--lookahead', dest='lookahead', action='store_true', help='use lookahead optimizer')
    parser.add_argument('--la-k', type=int, default=6, help='k of lookahead.')
    parser.add_argument('--la-alpha', type=float, default=0.5, help='alpha of lookahead.')
    parser.add_argument('--clip-grad', type=float, default=0., help='gradient clipping')

    # Checkpoints
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
    parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
    parser.add_argument('--resume', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='Number of batches between log messages')
    parser.add_argument('--test-on-attack', action='store_true', help='Run attack during the valudation too')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: random)')

    # Architecture
    parser.add_argument('--arch', type=str, default='resnet', choices=['resnet', 'wideresnet', 'lenet5'],
                        help='Network architecture. Wideresnet implementation taken from: https://github.com/yaodongyu/TRADES')
    parser.add_argument('--layers', type=int, default=20, metavar='L', help='Number of layers')
    parser.add_argument('--width', type=float, default=1, metavar='W', help='Width multiplier')
    parser.add_argument('--save_data_mode', type=str, default=None, help='set the csv file name for smoothed model outputs')
    parser.add_argument('--loader_type', type=str, default='val_loader', help='loader type - training or validation')

    # Attack
    parser.add_argument('--attack', default='pgd', type=str, metavar='ATT', help='attack type')
    parser.add_argument('--eps', default=8, type=int, metavar='EPS', help='epsilon of attack, will be divided by 255')
    parser.add_argument('--alpha', default=None, type=float, metavar='ALPHA', help='step of attack')
    parser.add_argument('--step-size', default=0.007, help='perturb step size')  # used in TRADES #TODO: remove
    parser.add_argument('--attack_k', default=7, type=int, metavar='ATTK', help='number of iterations for the attack')
    parser.add_argument('--beta', default=1.0, type=float, help='weight of adversarial loss, i.e., 1/lambda in TRADES')

    parser.add_argument('--targeted', dest='targeted', action='store_true', help='Use targeted attacks')
    parser.add_argument('--multi_att', dest='multi_att', action='store_true', help='test multiple attacks')
    parser.add_argument('--att-names', type=str, nargs='+',
                        default=['pgd' for x in range(22)],
                        help='names of multiple attacks.')
    parser.add_argument('--att-k-list', type=int, nargs='+',
                        default=[10 for x in range(22)],
                        help='number of iterations for multiple attacks.')
    parser.add_argument('--att-targeted-list', type=bool, nargs='+', default=[False for x in range(22)],
                        help='Use targeted attack for each of the multiple attacks.')

    # PNI
    parser.add_argument('--pni', dest='pni', action='store_true', help='Use PNI')
    parser.add_argument('--cpni', dest='cpni', action='store_true', help='Use colored PNI')
    parser.add_argument('--noise-decay', type=float, default=0, help='Weight decay (L2 penalty).')

    # Smoothing
    parser.add_argument('--smooth', default='', type=str, metavar='SMOOTH', help='smooth type')

    parser.add_argument('--noise_sd', type=float, default=0, metavar='SD',
                        help='noise standard variation for smooth model')
    parser.add_argument('--m_backward', type=int, default=8, metavar='NS',
                        help='number of monte carlo samples for epgd attacks')
    parser.add_argument('--m_forward', type=int, default=512, metavar='NS',
                        help='number of monte carlo samples for smooth model in forward')

    parser.add_argument('--test_noise', dest='test_noise', action='store_true',
                        help='test all specified options of Noise injection')
    parser.add_argument('--no_pred_prob', dest='no_pred_prob', action='store_true',
                        help='do not estimate the probability the predicted class is classified for a monte-carlo sample')
    parser.add_argument('--run_optimal', dest='run_optimal', action='store_true',
                        help='run optimal options of hyper params')

    parser.add_argument('--m-forward-list', type=int, nargs='+', default=[2 ** x for x in range(10)],
                        help='m_forward options to check while optimizing hyper params.')  #
    parser.add_argument('--noise-list', type=float, nargs='+', default=[x / 100 for x in range(51)],
                        help='noise_sd options to check while optimizing hyper params.')

    parser.add_argument('--weight-noise', dest='weight_noise', action='store_true', help='Use weight noise')
    parser.add_argument('--act-noise-a', dest='act_noise_a', action='store_true', help='Use activation noise A')
    parser.add_argument('--act-noise-b', dest='act_noise_b', action='store_true', help='Use activation noise B')  # TODO
    parser.add_argument('--noise-rank', type=int, default=5, metavar='R', help='Rank of colored noise')
    parser.add_argument('--weight-noise-d', type=float, default=0.25, metavar='NDS',
                        help='noise strength in the diagonal')
    parser.add_argument('--weight-noise-f', type=float, default=0.1, metavar='NDF',
                        help='noise strength outside the diagonal')

    parser.add_argument('--adv', dest='adv', action='store_true', help='Use adversarial training')
    parser.add_argument('--zero-start', dest='zero_start', action='store_true', help='Start from epoch 0')

    parser.add_argument('-g', '--gen-adv', dest='gen_adv', action='store_true', help='Generate adv samples')

    parser.add_argument('--experiment-name', default='exp1', type=str, help='experiment name')
    parser.add_argument('--no-norm', dest='no_norm', action='store_true', help='No normalization')
    parser.add_argument('--no-var-norm', dest='no_var_norm', action='store_true', help='Only normalize the mean')

    # transfer attack
    parser.add_argument('--transfer-attack', action='store_true', default=False, help='Use transfer attack')
    parser.add_argument('--attack-path', type=str, metavar='PATH', help='path to attack model (default: none)')
    parser.add_argument('--transfer-attack-noise', type=float, default=0, metavar='TASD',
                        help='noise standard variation for transfer attack smooth model')

    args = parser.parse_args()

    args.test_hyper_params = args.test_noise

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save == '':
        args.save = time_stamp
    else:
        args.save += '_time_' + time_stamp
    args.save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.gpus is not None and torch.cuda.is_available():
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.enabled = True
        cudnn.benchmark = True
        args.device = 'cuda:' + str(args.gpus[0])
        torch.cuda.set_device(args.gpus[0])
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = 'cpu'

    if args.type == 'float64':
        args.dtype = torch.float64
    elif args.type == 'float32':
        args.dtype = torch.float32
    elif args.type == 'float16':
        args.dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    args.mcpredict, args.mcepredict = False, False
    if args.smooth == 'mcpredict':
        args.mcpredict = True
    elif args.smooth == 'mcepredict':
        args.mcepredict = True
    elif args.smooth != '':
        raise ValueError('Wrong smooth method!')

    args.smoothing = args.mcpredict or args.mcepredict

    args.dataset_name = args.dataset
    if args.dataset == 'cifar10':
        args.dataset = torchvision.datasets.CIFAR10
        args.num_classes = 10
        from data_loaders.cifar_data import get_loaders
    elif args.dataset == 'cifar100':
        args.dataset = torchvision.datasets.CIFAR100
        args.num_classes = 100
        from data_loaders.cifar_data import get_loaders
    else:
        raise ValueError('Wrong dataset!')
    args.get_loaders = get_loaders

    if args.pni:
        args.noise_rank = 0
        args.weight_noise_f = 0
        if not args.act_noise_a and not args.act_noise_b:
            args.weight_noise = True
    elif args.cpni:
        if not args.act_noise_a and not args.act_noise_b:
            args.weight_noise = True
    else:
        args.noise_rank = 0
        args.weight_noise_d = 0
        args.weight_noise_f = 0
        args.weight_noise = False
        args.act_noise_a = False
        args.act_noise_b = False

    args.adv_w = args.beta / (args.beta + 1)

    if not args.smoothing:
        args.noise_sd = 0

    from models.resnet_cpni_smooth_inference import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, \
        resnet110_cifar, resnet164_cifar

    if args.arch == 'resnet':
        name_dict = {20: 'resnet20_cifar', 32: 'resnet32_cifar', 44: 'resnet44_cifar', 56: 'resnet56_cifar',
                     110: 'resnet110_cifar',164: 'resnet164_cifar', 50: 'resnet50'}
        net_dict = {20: resnet20_cifar, 32: resnet32_cifar, 44: resnet44_cifar, 56: resnet56_cifar,
                    110: resnet110_cifar, 164: resnet164_cifar}
        args.net = net_dict[args.layers]
        args.net_name = name_dict[args.layers]
    elif args.arch == 'wideresnet':
        name_dict = {28: 'wideresnet28', 34: 'wideresnet34'}
        net_dict = {28: wideresnet28, 34: wideresnet34}
        args.net = net_dict[args.layers]
        args.net_name = name_dict[args.layers]
    else:
        raise ValueError('Wrong architecture!')

    attack_dict = {'pgd': PGD, 'epgd': EPGD}
    args.attack_name = args.attack
    args.attack = attack_dict[args.attack]

    args.eps = args.eps / 255
    if args.alpha is None or args.alpha * args.attack_k < args.eps:
        new_alpha = args.eps / args.attack_k
        print("Alpha={} is too small For {} attack with Epsilon={} and k={}, fixing to {}".format(
            args.alpha, args.attack_name, args.eps, args.attack_k, new_alpha))
        args.alpha = new_alpha

    args.attacks = [args.attack]
    args.attacks_type = [args.attack_name]
    args.attacks_eps = [args.eps]
    args.attacks_alpha = [args.alpha]
    args.attacks_tar = [args.targeted]
    args.attacks_k = [args.attack_k]
    if args.multi_att:
        args.attacks_type = args.att_names
        args.attacks_eps = ([eps/255 for eps in args.eps_list] * len(args.att_names))[0:len(args.att_names)]
        args.attacks_k = (args.att_k_list * len(args.att_names))[0:len(args.att_names)]
        args.attacks_tar = (args.att_targeted_list * len(args.att_names))[0:len(args.att_names)]
        args.attacks = []
        args.attacks_alpha = []
        for i, att_name in enumerate(args.att_names):
            att = attack_dict[att_name]
            args.attacks.append(att)
            att_alpha = args.alpha
            if att_alpha is None or att_alpha * args.attacks_k[i] < args.attacks_eps[i]:
                att_alpha = args.attacks_eps[i] / args.attacks_k[i]
                print("Alpha={} is too small For {} attack with Epsilon={} and k={}, fixing to {}".format(
                    args.alpha, args.attacks_type[i], args.attacks_eps[i], args.attacks_k[i], att_alpha))
            args.attacks_alpha.append(att_alpha)

    args.attacks_add_params = [{} for att in args.attacks]
    for att_idx, att_name in enumerate(args.attacks_type):
        if att_name == 'pgd':
            args.attacks_add_params[att_idx] = {'n_iter': args.attacks_k[att_idx], 'alpha': args.attacks_alpha[att_idx]}
        elif att_name == 'epgd':
            args.attacks_add_params[att_idx] = {'n_iter': args.attacks_k[att_idx], 'alpha': args.attacks_alpha[att_idx]}
        else:
            raise ValueError('Wrong attack!')

    args.nesterov_momentum = (args.momentum != 0.0)

    print("args")
    print(args)

    csv_logger = CsvLogger(filepath=args.save_path, data=[])
    csv_logger.save_params(sys.argv, args)

    return args


def get_logger(args, name='CPNI'):
    # LOGGING SETUP
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # file handler
    fh = logging.FileHandler(os.path.join(args.save_path, args.experiment_name + '.log'))
    fh.setLevel(logging.DEBUG)
    # console handler with a higher log level (TODO: tqdm)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # put generic info on args
    logger.debug('{}'.format(args))
    logger.info('Seed = {}'.format(args.seed))
    logger.info('Device, dtype = {}, {}'.format(args.device, args.dtype))
    return logger


def log_noise_level(writer, layer, cnt, epoch):
    try:
        mean = torch.mean(torch.abs(layer.alphad_w))
        std = torch.std(torch.abs(layer.alphad_w))
        writer.add_scalar('zz_alphas/low_wd' + str(cnt), mean - std, epoch)
        writer.add_scalar('zz_alphas/mean_wd' + str(cnt), mean, epoch)
        writer.add_scalar('zz_alphas/high_wd' + str(cnt), mean + std, epoch)
    except:
        pass
    try:
        mean = torch.mean(torch.abs(layer.alphaf_w))
        std = torch.std(torch.abs(layer.alphaf_w))
        writer.add_scalar('zz_alphas/low_wf' + str(cnt), mean - std, epoch)
        writer.add_scalar('zz_alphas/mean_wf' + str(cnt), mean, epoch)
        writer.add_scalar('zz_alphas/high_wf' + str(cnt), mean + std, epoch)
    except:
        pass

    try:
        mean = torch.mean(torch.abs(layer.alphad_i))
        std = torch.std(torch.abs(layer.alphad_i))
        writer.add_scalar('zz_alphas/low_wd' + str(cnt), mean - std, epoch)
        writer.add_scalar('zz_alphas/mean_wd' + str(cnt), mean, epoch)
        writer.add_scalar('zz_alphas/high_wd' + str(cnt), mean + std, epoch)
    except:
        pass
    try:
        mean = torch.mean(torch.abs(layer.alphaf_i))
        std = torch.std(torch.abs(layer.alphaf_i))
        writer.add_scalar('zz_alphas/low_af' + str(cnt), mean - std, epoch)
        writer.add_scalar('zz_alphas/mean_af' + str(cnt), mean, epoch)
        writer.add_scalar('zz_alphas/high_af' + str(cnt), mean + std, epoch)
    except:
        pass
