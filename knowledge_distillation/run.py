import torchvision
import torch
from knowledge_distillation.soft_target_KD import SoftTargetKD
from models.wideresnet import wideresnet28
from data_loaders.cifar_data import get_loaders
from util.cross_entropy import CrossEntropyLoss
import pandas as pd
import knowledge_distillation.teacher_data as td

# TODO: import teacher model

def main():
    # TODO check and get which loss function the KD need (cross entropy?)
    # TODO define temperature (float)
    # TODO define distil weight (float)
    # TODO define device
    # TODO define log (true)
    # TODO define logdir

    student_model = wideresnet28()
    workers = 4
    train_loader, test_loader = get_loaders(dataset=torchvision.datasets.CIFAR10, data="./data", batch_size=256,
                                            val_batch_size=256, workers=workers)

    # TODO: extract it?
    args = pd.DataFrame({"momentum": 0.9, "learning_rate": 0.1, "nesterov_momentum": True, "decay": 0.001,
                         "temperature": 5, "distil_weight": 0.5, "device": 'cuda',
                         "log_dir": 'C:/Users/Yael/DL/KD_PROJECT/logs'})

    # TODO make sure that this sgd optimizer is good
    # TODO complete teacher optimizer
    optimizer_teacher = torch.optim.SGD(
        student_model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=args.nesterov_momentum)

    optimizer_student = torch.optim.SGD(
        student_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=args.nesterov_momentum)

    # initialize SoftTargetKD object
    soft_target_KD = SoftTargetKD(
        teacher_model=student_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer_teacher=optimizer_teacher,
        optimizer_student=optimizer_student,
        loss_fn=CrossEntropyLoss(),
        temp=args.temperature,
        distil_weight=args.distil_weight,
        device=args.device,
        log=True,
        logdir=args.log_dir
    )

    # SoftTargetKD.train_student

    # test student model

    # ======================

    # adversarial training............



# def teacher_data_loader(df):
#     for batch_index in range(10):
#         batch_tensors = df.loc[df['batch_number'] == batch_index]
#         # ........

if __name__ == '__main__':
    # main()

    torch.set_printoptions(threshold=10_000)
    teacher_data = td.TeacherData(data_dic={'clean_data': True, 'perturb_data': False}, m_forward=8)
    temp_batch_index = 0
    batch_teacher_out = teacher_data.get_predictions(mode='clean', batch_index=temp_batch_index)

    print(batch_teacher_out)


