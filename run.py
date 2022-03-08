import os
import shutil

import matplotlib
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.utils
from torch.distributions.dirichlet import Dirichlet
from tqdm import tqdm

from layers import get_noise_norm
from models.wideresnet import wideresnet28

matplotlib.use('Agg')


def transform_checkpoint(cp):
    new_cp = {}
    for entry in cp:
        new_name = entry.replace('module.', '')
        if new_name.startswith('1.'):
            new_name = new_name[2:]
        new_cp[new_name] = cp[entry]
    return new_cp


def train(model, loader, epoch, optimizer, criterion, writer, iter, experiment_name, logger, device, dtype, batch_size,
          log_interval, clip_grad=0.):
    model.train()
    correct1, correct5 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        loss, output = model.forward_backward(data, target, criterion)
        if clip_grad > 1e-12:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        corr, _, _, _ = correct(model, data, output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        writer.add_scalars('batch/train_loss', {experiment_name: loss.item()}, iter)
        writer.add_scalars('batch/top1', {experiment_name: corr[0] / data.shape[0]}, iter)
        writer.add_scalars('batch/top5', {experiment_name: corr[1] / data.shape[0]}, iter)
        iter += 1
    writer.add_scalars('epoch/top1', {experiment_name + "_train": correct1 / len(loader.sampler)}, epoch)
    writer.add_scalars('epoch/top5', {experiment_name + "_train": correct5 / len(loader.sampler)}, epoch)
    logger.debug(
        'Train Epoch: {}\tLoss: {:.6f}. '
        'Top-1 accuracy: {:.2f}%. '
        'Top-5 accuracy: {:.2f}%.'.format(epoch, loss.item(), 100 * correct1 / len(loader.sampler),
                                          100 * correct5 / len(loader.sampler)))
    try:
        size = len(loader.dataset)
    except:
        size = len(loader) * loader[0][0].shape[0]  # TODO
    return iter, loss.item(), correct1 / size, correct5 / size


def adv_train(model, loader, epoch, optimizer, criterion, writer, iter, experiment_name, logger, device, dtype,
              batch_size, log_interval, att, eps, adv_w, normalize, clip_grad=0., noise_decay=0.):
    model.train()
    correct1, correct5 = 0, 0
    correct1a, correct5a = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        att.model = model  # TODO
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        loss, loss_a, output, output_a = model.adv_forward_backward(data, target, criterion, att, eps, normalize, adv_w)
        if clip_grad > 1e-12:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        if noise_decay > 1e-12:
            mn = get_noise_norm(model, device, dtype)
            (mn * noise_decay).backward()

        optimizer.step()

        corr, _, _, _ = correct(model, data, output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]
        corra, _, _, _ = correct(model, data, output_a, target, topk=(1, 5))
        correct1a += corra[0]
        correct5a += corra[1]

        writer.add_scalars('batch/train_loss',
                           {experiment_name: loss.item(), experiment_name + "_adversarial": loss_a.item()}, iter)
        writer.add_scalars('batch/train_loss', {}, iter)
        writer.add_scalars('batch/top1', {experiment_name: corr[0] / data.shape[0],
                                          experiment_name + "_adversarial": corra[0] / data.shape[0]}, iter)
        writer.add_scalars('batch/top5', {experiment_name: corr[1] / data.shape[0],
                                          experiment_name + "_adversarial": corra[1] / data.shape[0]}, iter)
        iter += 1
    writer.add_scalars('epoch/top1', {experiment_name + "_train": correct1 / len(loader.dataset)}, epoch)
    writer.add_scalars('epoch/top5', {experiment_name + "_train": correct5 / len(loader.dataset)}, epoch)
    writer.add_scalars('epoch/top1', {experiment_name + "_train_adversarial": correct1a / len(loader.dataset)}, epoch)
    writer.add_scalars('epoch/top5', {experiment_name + "_train_adversarial": correct5a / len(loader.dataset)}, epoch)
    logger.debug(
        'Train Epoch: {}\tLoss: {:.6f}. '
        'Top-1 accuracy: {:.2f}%. '
        'Top-5 accuracy: {:.2f}%.'
        'Top-1 accuracy: {:.2f}%. '
        'Top-5 accuracy: {:.2f}%.'.format(epoch, loss.item(), 100 * correct1 / len(loader.dataset),
                                          100 * correct5 / len(loader.dataset), 100 * correct1a / len(loader.dataset),
                                          100 * correct5a / len(loader.dataset)))

    return iter, loss.item(), correct1 / len(loader.dataset), correct5 / len(loader.dataset)


def attack(model, loader, criterion, writer, iter, experiment_name, logger, epoch, att, eps, device, dtype,
           calc_prob=False, test_batch_idx=0):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0
    test_loss_a = 0
    correct1_a, correct5_a = 0, 0
    tested = 0
    rad, pred_prob, pred_prob_var = 0, 0, 0

    student_model = wideresnet28()
    student_model_path = 'knowledge_distillation/kd_models/student_20220227-195029.pt'
    student_model.load_state_dict(torch.load(student_model_path))
    student_model.to(device)
    att.model = student_model  # TODO: pass here the student model
    print(f"Loaded student model: {student_model_path}")

    for batch_idx, (data, target, image_indices) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        x_a, output, output_a, _ = att.perturb(data, target, eps)
        tl = criterion(output, target).item()
        test_loss += tl  # sum up batch loss
        corr, _, _, _ = correct(model, data, output, target, topk=(1, 5), mode='clean_data', batch_idx=batch_idx,
                                image_indices=image_indices)
        correct1 += corr[0]
        correct5 += corr[1]
        tla = criterion(output_a, target).item()
        test_loss_a += tla  # sum up batch loss
        corr_a, rad_batch, pred_prob_batch, pred_prob_var_batch = correct(model, x_a, output_a, target, topk=(1, 5),
                                                                          calc_prob=calc_prob, mode='perturb_data',
                                                                          batch_idx=batch_idx,
                                                                          image_indices=image_indices)
        correct1_a += corr_a[0]
        correct5_a += corr_a[1]
        rad += rad_batch
        pred_prob += pred_prob_batch
        pred_prob_var += pred_prob_var_batch
        if writer is not None:
            writer.add_scalars('batch/val_loss', {experiment_name: tl}, iter)
            writer.add_scalars('batch/val_loss', {experiment_name + "_adversarial": tla}, iter)
        iter += 1
        tested += data.shape[0]
        logger.debug("Batch {}/{}: Clean {}/{}, {:.3f}%; "
                     "Adv {}/{}, {:.3f}%".format(batch_idx + 1, len(loader), int(correct1),
                                                 tested,
                                                 correct1 / (tested) * 100., int(correct1_a),
                                                 tested,
                                                 correct1_a / (tested) * 100.))

    test_loss /= len(loader)
    test_loss_a /= len(loader)
    rad /= len(loader)
    pred_prob /= len(loader)
    pred_prob_var /= len(loader)
    if writer is not None:
        writer.add_scalars('epoch/loss', {experiment_name + "_val": test_loss}, epoch)
        writer.add_scalars('epoch/top1', {experiment_name + "_val": correct1 / len(loader.dataset)}, epoch)
        writer.add_scalars('epoch/top5', {experiment_name + "_val": correct5 / len(loader.dataset)}, epoch)
        writer.add_scalars('epoch/loss', {experiment_name + "_val_adversarial": test_loss_a}, epoch)
        writer.add_scalars('epoch/top1', {experiment_name + "_val_adversarial": correct1_a / len(loader.dataset)},
                           epoch)
        writer.add_scalars('epoch/top5', {experiment_name + "_val_adversarial": correct5_a / len(loader.dataset)},
                           epoch)
    logger.debug(
        'Test set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(test_loss, int(correct1), len(loader.dataset),
                                       100. * correct1 / len(loader.dataset), int(correct5),
                                       len(loader.dataset), 100. * correct5 / len(loader.dataset)))
    logger.debug(
        'Adverserial set (eps={}): Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(eps, test_loss_a, int(correct1_a), len(loader.dataset),
                                       100. * correct1_a / len(loader.dataset), int(correct5_a),
                                       len(loader.dataset), 100. * correct5_a / len(loader.dataset)))

    logger.debug(
        'Adverserial set variance (eps={}): Certefication radius: {}, Prominent Class Probability: {}, '
        'Prominent Class Variance: {}'.format(eps, rad, pred_prob, pred_prob_var))

    return iter, test_loss, correct1 / len(loader.dataset), correct5 / len(loader.dataset), \
           test_loss_a, correct1_a / len(loader.dataset), correct5_a / len(
        loader.dataset), rad, pred_prob, pred_prob_var


def targeted_attack(model, loader, criterion, writer, iter, experiment_name, logger, epoch, att, eps, num_classes,
                    device, dtype, calc_prob=False):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0
    test_loss_a = 0
    correct1_a, correct5_a = 0, 0
    rad, pred_prob, pred_prob_var = 0, 0, 0
    att.model = model  # TODO

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        x_a, output, output_a, _ = att.perturb(data, target, eps)

        tl = criterion(output, target).item()
        test_loss += tl  # sum up batch loss
        corr, _, _, _ = correct(model, data, output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        tla_targets = 0
        corr1_a_targets = 0
        corr5_a_targets = 0
        rad_targets = 0
        pred_prob_targets = 0
        pred_prob_var_targets = 0
        att_target = torch.zeros_like(target)
        for t in range(num_classes):
            att_target[0] = t
            x_a, output, output_a, _ = att.perturb(data, att_target, eps, target=True)
            tla = criterion(output_a, target).item()
            tla_targets += tla  # sum up targeted attacks
            corr_a_target, rad_target, pred_prob_target, pred_prob_var_target \
                = correct(model, x_a, output_a, target, topk=(1, 5), calc_prob=calc_prob)
            corr1_a_targets += corr_a_target[0]
            corr5_a_targets += corr_a_target[1]
            rad_targets += rad_target
            pred_prob_targets += pred_prob_target
            pred_prob_var_targets += pred_prob_var_target

        test_loss_a += tla_targets / num_classes  # sum up batch loss
        correct1_a += corr1_a_targets / num_classes
        correct5_a += corr5_a_targets / num_classes
        rad += rad_targets / num_classes
        pred_prob += pred_prob_targets / num_classes
        pred_prob_var += pred_prob_var_targets / num_classes

        if writer is not None:
            writer.add_scalars('batch/val_loss', {experiment_name: tl}, iter)
            writer.add_scalars('batch/val_loss', {experiment_name + "_adversarial": tla}, iter)
        iter += 1

    test_loss /= len(loader)
    test_loss_a /= len(loader)
    rad /= len(loader)
    pred_prob /= len(loader)
    pred_prob_var /= len(loader)
    if writer is not None:
        writer.add_scalars('epoch/loss', {experiment_name + "_val": test_loss}, epoch)
        writer.add_scalars('epoch/top1', {experiment_name + "_val": correct1 / len(loader.dataset)}, epoch)
        writer.add_scalars('epoch/top5', {experiment_name + "_val": correct5 / len(loader.dataset)}, epoch)
        writer.add_scalars('epoch/loss', {experiment_name + "_val_adversarial": test_loss_a}, epoch)
        writer.add_scalars('epoch/top1', {experiment_name + "_val_adversarial": correct1_a / len(loader.dataset)},
                           epoch)
        writer.add_scalars('epoch/top5', {experiment_name + "_val_adversarial": correct5_a / len(loader.dataset)},
                           epoch)
    logger.debug(
        '\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(test_loss, int(correct1), len(loader.dataset),
                                       100. * correct1 / len(loader.dataset), int(correct5),
                                       len(loader.dataset), 100. * correct5 / len(loader.dataset)))
    logger.debug(
        'Adverserial set (eps={}): Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(eps, test_loss_a, int(correct1_a), len(loader.dataset),
                                       100. * correct1_a / len(loader.dataset), int(correct5_a),
                                       len(loader.dataset), 100. * correct5_a / len(loader.dataset)))
    logger.debug(
        'Adverserial set variance (eps={}): Certefication radius: {}, Prominent Class Probability: {}, '
        'Prominent Class Variance: {}'.format(eps, rad, pred_prob, pred_prob_var))

    return iter, test_loss, correct1 / len(loader.dataset), correct5 / len(loader.dataset), \
           test_loss_a, correct1_a / len(loader.dataset), correct5_a / len(
        loader.dataset), rad, pred_prob, pred_prob_var


def test(model, loader, criterion, writer, iter, experiment_name, logger, epoch, device, dtype):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
            output = model(data)
            tl = criterion(output, target).item()  # sum up batch loss
            test_loss += tl
            corr, _, _, _ = correct(model, data, output, target, topk=(1, 5))
            correct1 += corr[0]
            correct5 += corr[1]

            writer.add_scalars('batch/val_loss', {experiment_name: tl}, iter)
            iter += 1

    test_loss /= len(loader)
    writer.add_scalars('epoch/loss', {experiment_name + "_val": test_loss}, epoch)
    writer.add_scalars('epoch/top1', {experiment_name + "_val": correct1 / len(loader.dataset)}, epoch)
    writer.add_scalars('epoch/top5', {experiment_name + "_val": correct5 / len(loader.dataset)}, epoch)
    logger.debug(
        '\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(test_loss, int(correct1), len(loader.dataset),
                                       100. * correct1 / len(loader.dataset), int(correct5),
                                       len(loader.dataset), 100. * correct5 / len(loader.dataset)))
    return iter, test_loss, correct1 / len(loader.dataset), correct5 / len(loader.dataset)


def correct(model, data, output, target, topk=(1,), calc_prob=False, mode=None, batch_idx=None, image_indices=None):
    """Computes the correct@k for the specified values of k"""

    # time_stamp_start = datetime.now()
    maxk = max(topk)
    pred, pred_prob, pred_prob_var = model.predict(data, output, maxk, calc_prob=calc_prob, save_data_mode=mode,
                                                   batch_idx=batch_idx, image_indices=image_indices)
    radius = -1
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        res.append(correct_k)

    return res, radius, pred_prob, pred_prob_var


def save_checkpoint(state, is_best1, filepath='./', filename='checkpoint.pth.tar', is_best2=False):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    best2_path = os.path.join(filepath, 'model_best2.pth.tar')
    torch.save(state, save_path)
    if is_best1:
        shutil.copyfile(save_path, best_path)
    if is_best2:
        shutil.copyfile(save_path, best2_path)
