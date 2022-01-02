import torchvision
import torch
from knowledge_distillation.soft_target_KD import SoftTargetKD
from models.wideresnet import wideresnet28
from data_loaders.cifar_data import get_loaders
from util.cross_entropy import CrossEntropyLoss
import pandas as pd
import knowledge_distillation.teacher_data as td


def main():
    student_model = wideresnet28()
    teacher_data = td.TeacherData(data_dic={'clean_data': True, 'perturb_data': False},
                                  m_forward=512)
    workers = 4
    train_loader, test_loader, _ = get_loaders(dataset=torchvision.datasets.CIFAR10,
                                               data="./data",
                                               batch_size=256,
                                               val_batch_size=256,
                                               workers=workers)

    # TODO: extract it?
    args = pd.DataFrame({'momentum': 0.9,
                         'learning_rate': 0.1,
                         'nesterov_momentum': True,
                         'decay': 0.001,
                         'temperature': 5,
                         'distil_weight': 0.5,
                         'device': 'cpu',
                         'log_dir': 'C:/Users/Yael/DL/KD_PROJECT/logs'}, index=[0])

    optimizer_student = torch.optim.SGD(
        student_model.parameters(),
        args.learning_rate[0],
        momentum=args.momentum[0],
        weight_decay=args.decay[0],
        nesterov=args.nesterov_momentum[0])

    # initialize SoftTargetKD object
    soft_target_KD = SoftTargetKD(
        teacher_data=teacher_data,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer_student=optimizer_student,
        loss_fn=CrossEntropyLoss(),
        temp=args.temperature[0],
        distil_weight=args.distil_weight[0],
        device=args.device[0],
        log=True,
        logdir=args.log_dir[0]
    )

    soft_target_KD.train_student()


if __name__ == '__main__':
    main()

    # torch.set_printoptions(threshold=10_000)
    # teacher_data1 = td.TeacherData(data_dic={'clean_data': True, 'perturb_data': False}, m_forward=512)
    # temp_image_indices = list(range(255))
    # batch_teacher_out = teacher_data1.get_predictions_by_image_indices(mode='clean', image_indices=temp_image_indices)
    # print(batch_teacher_out)


