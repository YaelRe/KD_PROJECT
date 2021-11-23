import csv
import os
import sys

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from logger import CsvLogger
from run import test, train, save_checkpoint, adv_train, attack
from smoothing.smooth import Smooth
from util.cross_entropy import CrossEntropyLoss
from util.lookahead import RAdam, Lookahead
from utils import get_args, log_noise_level, get_logger


def main():
    args = get_args()
    device, dtype = args.device, args.dtype
    add_args = {'weight_noise': args.weight_noise, 'act_noise_a': args.act_noise_a, 'act_noise_b': args.act_noise_b,
                'rank': args.noise_rank, 'noised_strength': args.weight_noise_d, 'noisef_strength': args.weight_noise_f,
                'num_classes': args.num_classes}
    if args.dataset == torchvision.datasets.ImageNet:
        add_args['pretrained'] = True
    else:
        add_args['width'] = args.width
    add_args['num_classes'] = args.num_classes
    smoothing_args = {}
    if args.smoothing:
        smoothing_args = {'noise_sd': args.noise_sd, 'm_forward': args.m_forward, 'm_backward': args.m_backward,
                          'm_certify': args.m_certify, 'smooth': args.smooth, 'sample_anti_adv': args.sample_anti_adv,
                          'anti_adv_att': args.anti_adv_att, 'anti_adv_eps': args.sample_eps}

    if args.no_norm:
        smoothing_args['normalization'] = None
    else:
        smoothing_args['normalization'] = args.dataset_name

    model = Smooth(args.net(**add_args), **smoothing_args)

    logger = get_logger(args, name='CPNI EPGD train')
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.debug(model)
    logger.info("Number of parameters {}".format(num_parameters))

    train_loader, val_loader, adv_data = args.get_loaders(args.dataset, args.data, args.batch_size, args.val_batch_size,
                                                          args.workers)
    # define loss function (criterion) and optimizer
    criterion = CrossEntropyLoss()

    best_test, best_adv = 0, 0

    model, criterion = model.to(device=device, dtype=dtype), criterion.to(device=device, dtype=dtype)
    if args.gpus is not None:
        model = model.data_parallel(args.gpus)

    # optionally resume from a checkpoint
    data = None
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            try:
                best_adv = checkpoint['best_adv1']
            except:
                pass
            model.load_state_dict(checkpoint['state_dict'])
            opt_state = checkpoint['optimizer']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')
            csv_path = os.path.join(args.resume, 'results.csv')
            logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            try:
                best_adv = checkpoint['best_adv1']
            except:
                pass
            model.load_state_dict(checkpoint['state_dict'])
            opt_state = checkpoint['optimizer']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            logger.error("=> no checkpoint found at '{}'".format(args.resume))

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.decay, nesterov=True)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.decay)
    elif args.opt == 'radam':
        optimizer = RAdam(model.parameters(), args.learning_rate, weight_decay=args.decay)
    else:
        raise ValueError('Wrong optimzier!')

    if args.lookahead:
        optimizer = Lookahead(optimizer, k=args.la_k, alpha=args.la_alpha)
    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
    if args.zero_start:
        args.start_epoch = 0
        best_test, best_adv = 0, 0
    else:
        try:
            optimizer.load_state_dict(opt_state)
        except:
            pass

    if args.evaluate:
        loss, top1, top5 = test(model, val_loader, criterion, device, dtype)  # TODO
        return

    csv_logger = CsvLogger(filepath=args.save_path, data=data)
    csv_logger.save_params(sys.argv, args)

    writer = SummaryWriter(log_dir=args.save_path)
    claimed_acc1 = None
    claimed_acc5 = None
    if args.adv:
        normalize = {'mean': np.array([0.491, 0.482, 0.447]), 'std': np.array([0.247, 0.243, 0.262])}
        a = smoothing_args
        a.update(add_args)
        a['width'] = args.width
        adv_train_network(args.start_epoch, args.epochs, scheduler, model, train_loader, val_loader, optimizer,
                          criterion, device, dtype, args.batch_size, args.log_interval, writer, logger,
                          args.experiment_name, args.save_path, claimed_acc1, claimed_acc5, best_test, best_adv,
                          args.attack, args.eps, args.adv_w, normalize, args.test_on_attack, args, a,
                          mode=args.adv_mode)

    else:
        train_network(args.start_epoch, args.epochs, scheduler, model, train_loader, val_loader, adv_data, optimizer,
                      criterion, device, dtype, args.batch_size, args.log_interval, writer, logger,
                      args.experiment_name, args.save_path, claimed_acc1, claimed_acc5, best_test)


def train_network(start_epoch, epochs, scheduler, model, train_loader, val_loader, adv_data, optimizer, criterion,
                  device, dtype, batch_size, log_interval, writer, logger, experiment_name, save_path, claimed_acc1,
                  claimed_acc5, best_test):
    noisy_layers = []
    for layer in model.modules():
        from layers import NoisedConv2DColored
        if isinstance(layer, NoisedConv2DColored):
            noisy_layers.append(layer)
    cs_dict = {'alphas_wd': {}, 'alphas_wf': {}, 'alphas_ad': {}, 'alphas_af': {}}
    for i in range(len(noisy_layers)):
        cs_dict['alphas_wd']['layer_' + str(i + 1)] = ['Margin', ['zz_alphas/mean_wd' + str(i + 1),
                                                                  'zz_alphas/low_wd' + str(i + 1),
                                                                  'zz_alphas/high_wd' + str(i + 1)]]
        cs_dict['alphas_wf']['layer_' + str(i + 1)] = ['Margin', ['zz_alphas/mean_wf' + str(i + 1),
                                                                  'zz_alphas/low_wf' + str(i + 1),
                                                                  'zz_alphas/high_wf' + str(i + 1)]]
        cs_dict['alphas_ad']['layer_' + str(i + 1)] = ['Margin', ['zz_alphas/mean_ad' + str(i + 1),
                                                                  'zz_alphas/low_ad' + str(i + 1),
                                                                  'zz_alphas/high_ad' + str(i + 1)]]
        cs_dict['alphas_af']['layer_' + str(i + 1)] = ['Margin', ['zz_alphas/mean_af' + str(i + 1),
                                                                  'zz_alphas/low_af' + str(i + 1),
                                                                  'zz_alphas/high_af' + str(i + 1)]]

    writer.add_custom_scalars(cs_dict)
    train_it, val_it = 0, 0
    for epoch in range(start_epoch, epochs + 1):
        train_it, train_loss, train_accuracy1, train_accuracy5 = train(model, train_loader, epoch, optimizer, criterion,
                                                                       writer, train_it, experiment_name, logger,
                                                                       device, dtype, batch_size, log_interval)
        val_it, test_loss, test_accuracy1, test_accuracy5 = test(model, val_loader, criterion, writer, val_it,
                                                                 experiment_name, logger, epoch, device, dtype)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.module.state_dict() if isinstance(model,
                                                                               nn.DataParallel) else model.state_dict(),
                         'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, test_accuracy1 > best_test, filepath=save_path)

        if test_accuracy1 > best_test:
            best_test = test_accuracy1
        cnt = 0
        for layer in model.modules():
            from layers import NoisedConv2DColored
            if isinstance(layer, NoisedConv2DColored):
                cnt += 1
                log_noise_level(writer, layer, cnt, epoch)
        scheduler.step()

    logger.info('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))


def adv_train_network(start_epoch, epochs, scheduler, model, train_loader, val_loader, optimizer, criterion, device,
                      dtype, batch_size, log_interval, writer, logger, experiment_name, save_path, claimed_acc1,
                      claimed_acc5, best_test, best_adv, adv_method, eps, adv_w, normalize, test_on_attack, args,
                      subts_args=None, mode='None'):
    noisy_layers = []
    for layer in model.modules():
        from layers import NoisedConv2DColored
        if isinstance(layer, NoisedConv2DColored):
            noisy_layers.append(layer)
    cs_dict = {'alphas_wd': {}, 'alphas_wf': {}, 'alphas_ad': {}, 'alphas_af': {}}
    for i in range(len(noisy_layers)):
        cs_dict['alphas_wd']['layer_' + str(i + 1)] = ['Margin', ['zz_alphas/mean_wd' + str(i + 1),
                                                                  'zz_alphas/low_wd' + str(i + 1),
                                                                  'zz_alphas/high_wd' + str(i + 1)]]
        cs_dict['alphas_wf']['layer_' + str(i + 1)] = ['Margin', ['zz_alphas/mean_wf' + str(i + 1),
                                                                  'zz_alphas/low_wf' + str(i + 1),
                                                                  'zz_alphas/high_wf' + str(i + 1)]]
        cs_dict['alphas_ad']['layer_' + str(i + 1)] = ['Margin', ['zz_alphas/mean_ad' + str(i + 1),
                                                                  'zz_alphas/low_ad' + str(i + 1),
                                                                  'zz_alphas/high_ad' + str(i + 1)]]
        cs_dict['alphas_af']['layer_' + str(i + 1)] = ['Margin', ['zz_alphas/mean_af' + str(i + 1),
                                                                  'zz_alphas/low_af' + str(i + 1),
                                                                  'zz_alphas/high_af' + str(i + 1)]]

    writer.add_custom_scalars(cs_dict)
    att_object = adv_method(model, criterion, **args.attacks_add_params[0])
    train_it, val_it = 0, 0

    for epoch in range(start_epoch, epochs + 1):

        if mode == 'None':
            train_it, train_loss, train_accuracy1, train_accuracy5 = adv_train(model, train_loader, epoch, optimizer,
                                                                               criterion, writer, train_it,
                                                                               experiment_name,
                                                                               logger, device, dtype, batch_size,
                                                                               log_interval, att_object, eps, adv_w,
                                                                               normalize, clip_grad=args.clip_grad,
                                                                               noise_decay=args.noise_decay)
        if test_on_attack:
            val_it, test_loss, test_accuracy1, test_accuracy5, \
            adv_loss, adv_accuracy1, adv_accuracy5, _ = attack(model, val_loader, criterion, writer, val_it,
                                                               experiment_name, logger, epoch, att_object, eps, device,
                                                               dtype)
            save_checkpoint(
                {'epoch': epoch + 1,
                 'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                 'best_prec1': best_test, 'best_adv1': best_adv,
                 'optimizer': optimizer.state_dict()}, test_accuracy1 > best_test, filepath=save_path,
                is_best2=adv_accuracy1 > best_adv)
            if adv_accuracy1 > best_adv:
                best_adv = adv_accuracy1
        else:
            val_it, test_loss, test_accuracy1, test_accuracy5 = test(model, val_loader, criterion, writer, val_it,
                                                                     experiment_name, logger, epoch, device, dtype)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.module.state_dict() if isinstance(model,
                                                                                   nn.DataParallel) else model.state_dict(),
                             'best_prec1': best_test,
                             'optimizer': optimizer.state_dict()}, test_accuracy1 > best_test, filepath=save_path)

        if test_accuracy1 > best_test:
            best_test = test_accuracy1
        cnt = 0
        for layer in model.modules():
            from layers import NoisedConv2DColored
            if isinstance(layer, NoisedConv2DColored):
                cnt += 1
                log_noise_level(writer, layer, cnt, epoch)
        scheduler.step()

    logger.info('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))


if __name__ == '__main__':
    main()
