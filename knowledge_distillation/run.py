import torchvision
import torch
import pandas as pd
import random
import datetime
import argparse

from models.wideresnet import wideresnet28
from attacks import PGD, EPGD
from data_loaders.cifar_data import get_loaders
from util.cross_entropy import CrossEntropyLoss
import torch.nn.functional as F

from knowledge_distillation.kd.soft_target_KD import SoftTargetKD
import knowledge_distillation.kd.teacher_data as td

parser = argparse.ArgumentParser(description='KD Training')
parser.add_argument('clean_train_data', type=bool)
parser.add_argument('perturb_train_data', type=bool)
parser.add_argument('clean_test_data', type=bool)
parser.add_argument('perturb_test_data', type=bool)
parser.add_argument('log-dir', type=str, help='folder to save model and training log')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--temperature', default=2.0, type=float, metavar='T')
parser.add_argument('--distill-weight', default=0.5, type=float, metavar='DW')
parser.add_argument('--gpu', default=None, type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

#####################
# Attack params
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--attack', default='PGD', type=str, choices=['EPGD', 'PGD'])
parser.add_argument('--epsilon', default=64.0, type=float)
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

# TODO: EPGD-specific?

# args = parser.parse_args()
#
# args.epsilon /= 256.0
# args.init_norm_DDN /= 256.0

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    load_student_model = False  # TODO: load it from params file
    student_model = wideresnet28()

    if load_student_model:
        resume_path = 'models/student.pt'  # in local : './models/student.pt' in server: 'models/student.pt'
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')  # map_location=device
        # args.start_epoch = checkpoint['epoch'] - 1
        # student_model.load_state_dict(transform_checkpoint(checkpoint['state_dict']))
        student_model.load_state_dict(transform_checkpoint(checkpoint))  # TODO: make sure with Adina

    teacher_data = td.TeacherData(data_dic={'clean_train_data': True, 'clean_test_data': True,
                                            'perturb_train_data': False, 'perturb_test_data': False},
                                  m_forward=512)
    workers = 4
    train_loader, test_loader, _ = get_loaders(dataset=torchvision.datasets.CIFAR10,
                                               data="./data",
                                               batch_size=256,
                                               val_batch_size=256,
                                               workers=workers)

    # TODO: extract it?
    # 'student_loss': torch.nn.MSELoss(),
    # 'student_loss': F.cross_entropy,
    # CrossEntropyLoss()
    args = pd.DataFrame({'momentum': 0.9,
                         'learning_rate': 0.01,
                         'nesterov_momentum': True,
                         'decay': 0.0001,
                         'temperature': 2,
                         'distill_weight': 0.5,
                         'student_loss': torch.nn.MSELoss(),
                         'device': 'cuda',
                         'log_dir': 'knowledge_distillation/logs/' + current_time
                         }, index=[0])

    optimizer_student_SGD = torch.optim.SGD(
        student_model.parameters(),
        args.learning_rate[0],
        momentum=args.momentum[0],
        weight_decay=args.decay[0],
        nesterov=args.nesterov_momentum[0])
    optimizer_student_ADAM = torch.optim.Adam(
        student_model.parameters(),
        args.learning_rate[0],
        weight_decay=args.decay[0], )
    print(f'lr = {args.learning_rate[0]}')

    # att_object = PGD(student_model, args.student_loss[0], n_iter=2, alpha=0.006)
    att_object = None

    # initialize SoftTargetKD object
    soft_target_KD = SoftTargetKD(
        teacher_data=teacher_data,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer_student=optimizer_student_SGD,
        loss_fn=torch.nn.MSELoss(),
        temp=args.temperature[0],
        distil_weight=args.distill_weight[0],
        device=args.device[0],
        att_object=att_object,
        log=True,
        logdir=args.log_dir[0]
    )

    soft_target_KD.train_student(epochs=100)
    soft_target_KD.evaluate()
    soft_target_KD.evaluate_teacher()


def transform_checkpoint(cp):
    new_cp = {}
    for entry in cp:
        new_name = entry.replace('module.', '')
        if new_name.startswith('1.'):
            new_name = new_name[2:]
        new_cp[new_name] = cp[entry]
    return new_cp


if __name__ == '__main__':
    # random.seed(42)
    torch.set_printoptions(threshold=10_000)
    main()

    # student_model = torch.load('./results/student.pt', map_location=torch.device('cpu'))

    # torch.set_printoptions(threshold=10_000)
    # teacher_data1 = td.TeacherData(data_dic={'clean_data': True, 'perturb_data': False}, m_forward=512)
    # temp_image_indices = list(range(255))
    # batch_teacher_out = teacher_data1.get_predictions_by_image_indices(mode='clean', image_indices=temp_image_indices)
    # print(batch_teacher_out)
