import torchvision
import torch
import pandas as pd
import random

from models.wideresnet import wideresnet28
from data_loaders.cifar_data import get_loaders
from util.cross_entropy import CrossEntropyLoss

from knowledge_distillation.kd.soft_target_KD import SoftTargetKD
import knowledge_distillation.kd.teacher_data as td


def main():
    student_model = wideresnet28()
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
    args = pd.DataFrame({'momentum': 0.9,
                         'learning_rate': 0.001,
                         'nesterov_momentum': True,
                         'decay': 0.001,
                         'temperature': 2,
                         'distil_weight': 0.3,
                         'device': 'cuda',
                         'log_dir': 'knowledge_distillation/logs/'}, index=[0])

    optimizer_student = torch.optim.SGD(
        student_model.parameters(),
        args.learning_rate[0],
        momentum=args.momentum[0],
        weight_decay=args.decay[0],
        nesterov=args.nesterov_momentum[0])
    print(f'lr = {args.learning_rate[0]}')

    # initialize SoftTargetKD object
    soft_target_KD = SoftTargetKD(
        teacher_data=teacher_data,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer_student=optimizer_student,
        loss_fn=torch.nn.MSELoss(),
        temp=args.temperature[0],
        distil_weight=args.distil_weight[0],
        device=args.device[0],
        log=True,
        logdir=args.log_dir[0]
    )

    soft_target_KD.train_student(epochs=25)
    soft_target_KD.evaluate()
    soft_target_KD.evaluate_teacher()


if __name__ == '__main__':
    random.seed(42)
    main()

    # student_model = torch.load('./results/student.pt', map_location=torch.device('cpu'))


    # torch.set_printoptions(threshold=10_000)
    # teacher_data1 = td.TeacherData(data_dic={'clean_data': True, 'perturb_data': False}, m_forward=512)
    # temp_image_indices = list(range(255))
    # batch_teacher_out = teacher_data1.get_predictions_by_image_indices(mode='clean', image_indices=temp_image_indices)
    # print(batch_teacher_out)


