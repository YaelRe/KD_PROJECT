import torch.nn as nn
import torch.nn.functional as F

from knowledge_distillation.kd.KD_framework import KDFramework


class SoftTargetKD(KDFramework):
    """
    Original implementation of Knowledge distillation from the paper "Distilling the
    Knowledge in a Neural Network" https://arxiv.org/pdf/1503.02531.pdf
    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        teacher_data,
        student_model,
        train_loader,
        val_loader,
        optimizer_student,
        scheduler,
        loss_fn=nn.MSELoss(),
        temp=20.0,
        distill_weight=0.5,
        perturb_distill_weight=0.5,
        eps=8,
        device="cpu",
        att_object=None,
        attack_model=None,
        experiment_name='exp_name',
        log=False,
        logdir="./Experiments",
    ):
        super(SoftTargetKD, self).__init__(
            teacher_data,
            student_model,
            train_loader,
            val_loader,
            optimizer_student,
            scheduler,
            loss_fn,
            temp,
            distill_weight,
            perturb_distill_weight,
            eps,
            device,
            att_object,
            attack_model,
            experiment_name,
            log,
            logdir,
        )

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true, distil_weight):
        """
        Function used for calculating the KD loss during distillation
        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        soft_teacher_out = F.softmax(y_pred_teacher / self.temp, dim=1)
        soft_student_out = F.softmax(y_pred_student / self.temp, dim=1)

        loss = (1 - distil_weight) * F.cross_entropy(y_pred_student, y_true)
        loss += (distil_weight * self.temp * self.temp) * self.loss_fn(
            soft_teacher_out, soft_student_out
        )
        return loss
