import torchvision
import torch
import random
import datetime
import argparse

from models.wideresnet import wideresnet28
from attacks import PGD
from data_loaders.cifar_data import get_loaders
from util.cross_entropy import CrossEntropyLoss
from util.lookahead import Lookahead
from torch.optim.lr_scheduler import MultiStepLR

from knowledge_distillation.kd.soft_target_KD import SoftTargetKD
import knowledge_distillation.kd.teacher_data as td

parser = argparse.ArgumentParser(description='KD Training')
parser.add_argument('--hist_data', default=False, type=bool)
parser.add_argument('--soft_data', default=False, type=bool)
parser.add_argument('--log_dir', '--log-dir', type=str, default='knowledge_distillation/logs/',
                    help='folder to save model and training log')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nesterov_momentum', default=True, type=bool)
parser.add_argument('--weight-decay', '--weight_decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--m_forward', default=512, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--temperature', default=2.0, type=float, metavar='T')
parser.add_argument('--distill_weight', default=0.5, type=float, metavar='DW')
parser.add_argument('--perturb_distill_weight', default=0.5, type=float, metavar='PDW')
parser.add_argument('--loss', default='MSE', type=str)
parser.add_argument('--opt', default='SGD', type=str)
parser.add_argument('--gpu', default=None, type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--load-student-model', default=False, type=bool, help='load the student model from a file')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[100], help='Decrease learning rate at these epochs.')
parser.add_argument('--lookahead', default=False, type=bool, help='use lookahead optimizer')
parser.add_argument('--la-k', type=int, default=6, help='k of lookahead.')
parser.add_argument('--la-alpha', type=float, default=0.5, help='alpha of lookahead.')
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('--eval-teacher', default=False, type=bool)
parser.add_argument('--resume-path', type=str, metavar='PATH', help='path to model (default: none)')

#####################
# Attack params
parser.add_argument('--adv-training2', action='store_true')
parser.add_argument('--adv_training', default=False, type=bool)  # yael
parser.add_argument('--attack', default='PGD', type=str, choices=['EPGD', 'PGD'])
parser.add_argument('--epsilon', default=64.0, type=float)
parser.add_argument('--n_iter', default=10, type=int)
parser.add_argument('--alpha', default=0.006, type=int)
parser.add_argument('--num-steps', default=10, type=int)
parser.add_argument('--warmup', default=1, type=int, help="Number of epochs over which \
-                    the maximum allowed perturbation increases linearly from zero to args.epsilon.")
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
parser.add_argument('--train-multi-noise', action='store_true',
                    help="if included, the weights of the network are optimized using all the noise samples. \
-                       Otherwise, only one of the samples is used.")
parser.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")

# PGD-specific
parser.add_argument('--random-start', default=True, type=bool)

# Transfer attack
parser.add_argument('--transfer-attack', action='store_true', default=False, help='Use transfer attack')
parser.add_argument('--attack-path', type=str, metavar='PATH', help='path to attack model (default: none)')
parser.add_argument('--experiment-name', default='exp1', type=str, help='experiment name')


# args.epsilon /= 256.0
# args.init_norm_DDN /= 256.0

# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)


def main():
    args = parser.parse_args()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    load_student_model = args.load_student_model
    student_model = wideresnet28()

    if load_student_model:
        # resume_path = 'models/student.pt'  # in local : './models/student.pt' in server: 'models/student.pt'
        print("=> loading checkpoint '{}'".format(args.resume_path))
        checkpoint = torch.load(args.resume_path, map_location='cpu')  # map_location=device
        # args.start_epoch = checkpoint['epoch'] - 1
        # student_model.load_state_dict(transform_checkpoint(checkpoint['state_dict']))
        # student_model.load_state_dict(transform_checkpoint(checkpoint))
        student_model.load_state_dict(torch.load(args.attack_path))
        student_model.to(args.device)

    transfer_attack_obj = None
    attack_model = None
    target_model = None
    if args.transfer_attack:
        # if "tar" in resume_path:
        #     checkpoint = torch.load(resume_path, map_location=device)
        #     args.start_epoch = checkpoint['epoch'] - 1
        #     model.load_state_dict(transform_checkpoint(checkpoint['state_dict']))
        #     logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        # else:
        print(f"=> Loading attack model: {args.attack_path}")
        attack_model = wideresnet28()
        # checkpoint = torch.load(args.attack_path, map_location='cpu')  # map_location=device
        # attack_model.load_state_dict(transform_checkpoint(checkpoint))
        attack_model.load_state_dict(torch.load(args.attack_path))
        attack_model.to(args.device)

    teacher_data = td.TeacherData(data_dic={'hist_data': args.hist_data,
                                            'soft_data': args.soft_data},
                                  m_forward=args.m_forward)
    workers = args.workers
    train_loader, test_loader, _ = get_loaders(dataset=torchvision.datasets.CIFAR10,
                                               data="./data",
                                               batch_size=args.batch_size,
                                               val_batch_size=args.batch_size,
                                               workers=workers)

    # 'student_loss': F.cross_entropy,
    if args.loss == 'MSE':
        student_loss = torch.nn.MSELoss()
    elif args.loss == 'CrossEntropy':
        student_loss = CrossEntropyLoss()
    else:
        raise Exception("error: loss function wasn't selected")

    if args.opt == 'SGD':
        optimizer_student = torch.optim.SGD(
            student_model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov_momentum)
    elif args.opt == 'ADAM':
        optimizer_student = torch.optim.Adam(
            student_model.parameters(),
            args.lr,
            weight_decay=args.weight_decay, )
    else:
        raise Exception("error: student optimizer wasn't selected")

    if args.lookahead:
        optimizer_student = Lookahead(optimizer_student)
    scheduler = MultiStepLR(optimizer_student, milestones=args.schedule, gamma=args.gamma)

    print(f'lr = {args.lr}')
    print(f'student_loss = {student_loss}')
    print(f'student_optimizer = {optimizer_student}')
    print(f'distill_weight = {args.distill_weight}')
    print(f'temperature = {args.temperature}')
    print(f'adv_training = {str(args.adv_training)}')

    if args.adv_training:
        att_object = PGD(student_model, student_loss, n_iter=args.n_iter, alpha=args.alpha)
    elif args.transfer_attack:
        att_object = PGD(attack_model, student_loss, n_iter=args.n_iter, alpha=args.alpha)
    else:
        att_object = None

    # initialize SoftTargetKD object
    soft_target_KD = SoftTargetKD(
        teacher_data=teacher_data,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer_student=optimizer_student,
        scheduler=scheduler,
        loss_fn=student_loss,
        temp=args.temperature,
        distill_weight=args.distill_weight,
        perturb_distill_weight=args.perturb_distill_weight,
        device=args.device,
        att_object=att_object,
        attack_model=attack_model,
        experiment_name=args.experiment_name,
        log=True,
        logdir='knowledge_distillation/logs/' + current_time
    )

    if args.train:
        soft_target_KD.train_student(epochs=args.epochs,
                                     save_model=True,
                                     save_model_pth=f"knowledge_distillation/kd_models/student_{current_time}.pt")
    if args.eval_teacher:
        soft_target_KD.evaluate_teacher()

    if args.transfer_attack:
        soft_target_KD.transfer_attack()

    soft_target_KD.evaluate()


def transform_checkpoint(cp):
    new_cp = {}
    for entry in cp:
        new_name = entry.replace('module.', '')
        if new_name.startswith('1.'):
            new_name = new_name[2:]
        new_cp[new_name] = cp[entry]
    return new_cp


if __name__ == '__main__':
    random.seed(42)
    main()

    # student_model = torch.load('./results/student.pt', map_location=torch.device('cpu'))
    # torch.set_printoptions(threshold=10_000)
