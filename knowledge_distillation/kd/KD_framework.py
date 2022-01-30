import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from copy import deepcopy
import os


class KDFramework:
    """
    Basic implementation of a general Knowledge Distillation framework
    :param teacher_data (pandas DataFrame): Teacher data
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
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
            loss_fn=nn.KLDivLoss(),
            temp=20.0,
            distil_weight=0.5,
            device="cpu",
            att_object=None,
            log=False,
            logdir="./Experiments",
    ):

        self.teacher_data = teacher_data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_student = optimizer_student
        self.temp = temp
        self.distil_weight = distil_weight
        self.att_object = att_object
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer_train_student_loss = SummaryWriter(log_dir=logdir+"_train_student_loss")
            self.writer_train_student_acc = SummaryWriter(log_dir=logdir+"_train_student_acc")
            self.writer_val_student_acc = SummaryWriter(log_dir=logdir+"_val_student_acc")
            self.writer_val_student_teacher_acc = SummaryWriter(log_dir=logdir+"_val_student_teacher_acc")

        if device == "cpu":
            print('device == cpu')
            self.device = torch.device("cpu")
        elif device == "cuda":
            print('device == cuda')
            if torch.cuda.is_available():
                print('cuda is available')
                self.device = torch.device("cuda")
            else:
                print(
                    "Either an invalid device or CUDA is not available. Defaulting to CPU."
                )
                self.device = torch.device("cpu")

        if teacher_data:
            print('temp = {}, distil_weight = {}'.format(self.temp, self.distil_weight))
            # self.teacher_model = teacher_model.to(self.device)
        else:
            print("Warning!!! Teacher is NONE.")

        self.student_model = student_model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.ce_fn = nn.CrossEntropyLoss().to(self.device)

    def _train_student(
            self,
            epochs=10,
            plot_losses=True,
            save_model=True,
            save_model_pth="knowledge_distillation/kd_models/student.pt",
    ):
        """
        Function to train student model - for internal use only.
        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """

        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)  # TODO: make sure that 50000 is ok..
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Student...")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0

            # TODO: add perturbation to data.....
            for batch_index, (data, label, image_indices) in enumerate(tqdm(self.train_loader)):

                data = data.to(self.device)
                label = label.to(self.device)

                if self.att_object:
                    # ===== Adversarial training ===== #
                    x_a, output, student_out, _ = self.att_object.perturb(data, label, eps=8/255)
                    # data.requires_grad = True

                    # x_a.requires_grad = False
                    # print(torch.eq(data[0], x_a[0]))
                    # student_out = self.student_model(x_a)# TODO: maybe not necessary and we can use the output_a
                    student_out = self.student_model(data)
                    # if batch_index == 0:
                    #     print("requires_grad----->" + str(data.requires_grad))
                    #     print("requires_grad----->" + str(x_a.requires_grad))
                    #     print("_backward_hooks --------->" +str( data._backward_hooks))
                    #     print("_backward_hooks --------->" + str(x_a._backward_hooks))
                    #     print("_grad------>" + str(data._grad))
                    #     print("_grad------>" + str(x_a._grad))

                else:
                    # ===== Regular training ===== #
                    student_out = self.student_model(data)

                # TODO: understand what to do about clean vs perturb data
                teacher_out = self.teacher_data.get_predictions_by_image_indices(mode='clean_train',
                                                                                 image_indices=image_indices.tolist())
                teacher_out = teacher_out.to(self.device)

                loss = self.calculate_kd_loss(student_out, teacher_out, label)

                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss.item()

            epoch_acc = correct / length_of_dataset

            _, epoch_val_acc, epoch_val_teacher_acc = self._evaluate_model(self.student_model, verbose=True)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )
            if self.log:
                self.writer_train_student_loss.add_scalar("Training loss/Student", epoch_loss, ep)
                self.writer_train_student_acc.add_scalar("Training accuracy/Student", epoch_acc, ep)
                self.writer_val_student_acc.add_scalar("Validation accuracy/Student", epoch_val_acc, ep)
                self.writer_val_student_teacher_acc.add_scalar("Validation Teacher accuracy/Student", epoch_val_teacher_acc, ep)

            loss_arr.append(epoch_loss)
            print(
                "Epoch: {}, Loss: {}, Accuracy: {}".format(
                    ep + 1, epoch_loss, epoch_acc
                )
            )

        if self.log:
            self.writer_train_student_loss.close()
            self.writer_train_student_acc.close()
            self.writer_val_student_acc.close()
            self.writer_val_student_teacher_acc.close()
        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)

    def train_student(
            self,
            epochs=10,
            plot_losses=True,
            save_model=True,
            save_model_pth="./models/student.pt",
    ):
        """
        Function that will be training the student
        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self._train_student(epochs, plot_losses, save_model, save_model_pth)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations
        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def _evaluate_model(self, model, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.
        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        student_teacher_correct = 0
        outputs = []

        with torch.no_grad():
            for data, target, image_indices in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                student_output = model(data)

                teacher_output = self.teacher_data.get_predictions_by_image_indices(mode='clean_test',
                                                                                    image_indices=image_indices.tolist())
                teacher_output = teacher_output.to(self.device)
                if isinstance(student_output, tuple):
                    student_output = student_output[0]
                outputs.append(student_output)

                student_pred = student_output.argmax(dim=1, keepdim=True)
                correct += student_pred.eq(target.view_as(student_pred)).sum().item()

                teacher_pred = teacher_output.argmax(dim=1, keepdim=True)
                student_teacher_correct += student_pred.eq(teacher_pred.view_as(student_pred)).sum().item()

        accuracy = correct / length_of_dataset
        student_teacher_accuracy = student_teacher_correct / length_of_dataset

        if verbose:
            print("-" * 80)
            print("Validation Accuracy: {}".format(accuracy))
            print("Student Teacher Validation Accuracy: {}".format(student_teacher_accuracy))
        return outputs, accuracy, student_teacher_accuracy

    def evaluate(self):
        """
        Evaluate method for printing accuracies of the trained network
        :param teacher (bool): True if you want accuracy of the teacher network
        """

        model = deepcopy(self.student_model).to(self.device)
        _, accuracy, _ = self._evaluate_model(model)

        return accuracy

    def evaluate_teacher(self, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.
        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []

        with torch.no_grad():
            for _, target, image_indices in self.val_loader:
                target = target.to(self.device)
                output = self.teacher_data.get_predictions_by_image_indices(mode='clean_test',
                                                                            image_indices=image_indices.tolist())
                output = output.to(self.device)
                # output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / length_of_dataset

        if verbose:
            print("-" * 80)
            print("Teacher Validation Accuracy: {}".format(accuracy))
            # Teacher Validation Accuracy: 0.8868
        return outputs, accuracy

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print("Total parameters for the student network are: {}".format(student_params))

    def post_epoch_call(self, epoch):
        """
        Any changes to be made after an epoch is completed.
        :param epoch (int) : current epoch number
        :return            : nothing (void)
        """

        pass

    # TODO: Create a plot of the training and testing loss and accuracy using the event log created with self.writer
    # https://stackoverflow.com/questions/36700404/tensorflow-opening-log-data-written-by-summarywriter
    # https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
    def plot_results(self):
        # from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        # event_acc = EventAccumulator('/path/to/summary/folder')
        # event_acc.Reload()
        # # Show all tags in the log file
        # print(event_acc.Tags())
        #
        # # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
        # w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'))
        pass
