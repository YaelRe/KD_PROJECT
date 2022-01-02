import csv
import gc
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter

from attacks import PGD, EPGD
from run import attack, targeted_attack
from util.cross_entropy import CrossEntropyLoss
from utils import get_args, get_logger
from smoothing.smooth import Smooth as Smooth


def test_hyper(args, logger):
    device, dtype = args.device, args.dtype
    add_args = {'weight_noise': args.weight_noise, 'act_noise_a': args.act_noise_a, 'act_noise_b': args.act_noise_b,
                'rank': args.noise_rank, 'noised_strength': args.weight_noise_d, 'noisef_strength': args.weight_noise_f,
                'num_classes': args.num_classes, 'width': args.width}

    # args.smoothing = True
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    attacks_eps_255 = [None if eps is None else int(eps * 255) for eps in args.attacks_eps]
    eps_string = ["" if eps is None else "_Eps" + str(eps) + "/255" for eps in attacks_eps_255]
    k_string = ["" if k is None else "_K" + str(k) for k in args.attacks_k]
    trageted_string = ["" if targeted is False else "_Targeted" for targeted in args.attacks_tar]
    alpha_string = ["" if alpha is None else "_Alpha" + str(alpha) for alpha in args.attacks_alpha]

    att_names = [att_name + k_string[att_idx] + eps_string[att_idx] + alpha_string[att_idx] + trageted_string[att_idx] for att_idx, att_name in
                 enumerate(args.attacks_type)]
    print("att_names")
    print(att_names)
    noise_list = [args.noise_sd]
    if args.test_noise:
        noise_list = args.noise_list

    results = [[] for m_forward in args.m_forward_list]
    results_adv = [[[] for m_forward in args.m_forward_list] for att_name in att_names]
    results_rad = [[] for m_forward in args.m_forward_list]
    results_pred_prob = [[] for m_forward in args.m_forward_list]
    results_pred_prob_var = [[] for m_forward in args.m_forward_list]
    max_adv_accuracy_list = [-1 for att_name in att_names]
    max_adv_reg_accuracy_list = [-1 for att_name in att_names]
    max_adv_rad_list = [-1 for att_name in att_names]
    max_adv_pred_prob_list = [-1 for att_name in att_names]
    max_adv_pred_prob_var_list = [-1 for att_name in att_names]
    max_adv_accuracy_noise_list = [0 for att_name in att_names]
    max_adv_accuracy_iteration_list = [0 for att_name in att_names]
    fieldsnames = ['smooth', 'dataset', 'net', 'm_backward', 'm_forward', 'noise',
                   'attack_name', 'attack_eps', 'attack_k', 'attack_alpha', 'attack_targeted', 'pred_prob', 'pred_prob_var', 'top1', 'top1_adv']
    args.results_csv = os.path.join(args.save_path, 'test_hyper_results.csv')

    print("Optimizing over noise and m_forward")
    # for att_idx, att_name in enumerate(att_names):
    for m_forward_idx, m_forward in enumerate(args.m_forward_list):
        for noise_idx, noise in enumerate(noise_list):
            print("starting run attack with noise={} and m_forward={}".format(noise, m_forward))
            model, val_loader, criterion = init_hyper_params(args, add_args, noise, m_forward,
                                                                     logger, device, dtype)

            test_loss, accuracy1, accuracy5, cer_rad, pred_prob, pred_prob_var, \
            test_loss_a_list, accuracy1_a_list, accuracy5_a_list = run_attacks(args, model, val_loader, criterion,
                                                                               logger, device, dtype)

            # process results
            results[m_forward_idx].append(accuracy1)
            results_rad[m_forward_idx].append(cer_rad)
            results_pred_prob[m_forward_idx].append(pred_prob)
            results_pred_prob_var[m_forward_idx].append(pred_prob_var)
            for att_idx, att_name in enumerate(att_names):
                results_adv[att_idx][m_forward_idx].append(accuracy1_a_list[att_idx])

            print("finished run attacks with noise={} and m_forward={}".format(noise, m_forward))
            print("att_names")
            print(att_names)
            print("args.m_forward_list")
            print(args.m_forward_list)
            print("noise_list")
            print(noise_list)

            for att_idx, att_name in enumerate(att_names):
                for m_forward_out_idx in range(m_forward_idx + 1):
                    current_iteration = args.m_forward_list[m_forward_out_idx]
                    noise_argmax = np.argmax(results_adv[att_idx][m_forward_out_idx])
                    # noise_max = results_adv[att_idx][m_forward_out_idx][noise_argmax]
                    current_max_adv_accuracy = results_adv[att_idx][m_forward_out_idx][noise_argmax]
                    current_max_adv_reg_accuracy = results[m_forward_out_idx][noise_argmax]
                    current_max_adv_rad = results_rad[m_forward_out_idx][noise_argmax]
                    current_max_adv_pred_prob = results_pred_prob[m_forward_out_idx][noise_argmax]
                    current_max_adv_pred_prob_var = results_pred_prob_var[m_forward_out_idx][noise_argmax]
                    current_max_adv_accuracy_noise = noise_list[noise_argmax]
                    if current_max_adv_accuracy > max_adv_accuracy_list[att_idx]:
                        max_adv_accuracy_list[att_idx] = current_max_adv_accuracy
                        max_adv_reg_accuracy_list[att_idx] = current_max_adv_reg_accuracy
                        max_adv_rad_list[att_idx] = current_max_adv_rad
                        max_adv_pred_prob_list[att_idx] = current_max_adv_pred_prob
                        max_adv_pred_prob_var_list[att_idx] = current_max_adv_pred_prob_var
                        max_adv_accuracy_noise_list[att_idx] = current_max_adv_accuracy_noise
                        max_adv_accuracy_iteration_list[att_idx] = current_iteration

                        print("results[att_name={}][m_forward={}]".format(att_name,
                                                                                       current_iteration))
                        print(results[m_forward_out_idx])
                        print("results_adv[att_name={}][m_forward={}]".format(att_name,
                                                                                           current_iteration))
                        print(results_adv[att_idx][m_forward_out_idx])
                        print("results_rad[att_name={}][m_forward={}]".format(att_name,
                                                                                           current_iteration))
                        print(results_rad[m_forward_out_idx])
                        print("results_pred_prob[att_name={}][m_forward={}]".format(att_name,
                                                                                           current_iteration))
                        print(results_pred_prob[m_forward_out_idx])
                        print("results_pred_prob_var[att_name={}][m_forward={}]".format(att_name,
                                                                                           current_iteration))
                        print(results_pred_prob_var[m_forward_out_idx])
                    print("best result for this attack and m_forward is"
                          " adverserial_accuracy=" + str(current_max_adv_accuracy) +
                          " accuracy=" + str(current_max_adv_reg_accuracy) +
                          " radius=" + str(current_max_adv_rad) +
                          " prediction probability=" + str(current_max_adv_pred_prob) +
                          " prediction probability variance=" + str(current_max_adv_pred_prob_var) +
                          " and was recived for att_name=" + att_name +
                          ", m_forward=" + str(current_iteration) +
                          " and noise=" + str(current_max_adv_accuracy_noise))
                print("best result so far is"
                      " adverserial_accuracy=" + str(max_adv_accuracy_list[att_idx]) +
                      " accuracy=" + str(max_adv_reg_accuracy_list[att_idx]) +
                      " radius=" + str(max_adv_rad_list[att_idx]) +
                      " pred_probability=" + str(max_adv_pred_prob_list[att_idx]) +
                      " pred_probability_variance=" + str(max_adv_pred_prob_var_list[att_idx]) +
                      " and was recived for att_name=" + att_name +
                      ", m_forward=" + str(max_adv_accuracy_iteration_list[att_idx]) +
                      " and noise=" + str(max_adv_accuracy_noise_list[att_idx]))

            with open(args.results_csv, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=fieldsnames)
                writer.writeheader()
                for att_idx, att_lst in enumerate(results_adv):
                    for m_forward_out_idx, m_forward_lst in enumerate(att_lst):
                        for noise_out_idx, res_adv in enumerate(m_forward_lst):
                            data = {'smooth': args.smooth, 'dataset': args.dataset_name,
                                    'net': args.net_name, 'm_backward': args.m_backward,
                                    'm_forward': args.m_forward_list[m_forward_out_idx],
                                    'noise': noise_list[noise_out_idx],
                                    'attack_name': args.attacks_type[att_idx],
                                    'attack_eps': attacks_eps_255[att_idx],
                                    'attack_k': args.attacks_k[att_idx],
                                    'attack_targeted': args.attacks_tar[att_idx],
                                    'attack_alpha': args.attacks_alpha[att_idx],
                                    'top1': results[m_forward_out_idx][noise_out_idx],
                                    'pred_prob': results_pred_prob[m_forward_out_idx][
                                        noise_out_idx],
                                    'pred_prob_var': results_pred_prob_var[m_forward_out_idx][
                                        noise_out_idx],
                                    'top1_adv': res_adv}
                            writer.writerow(data)

            del model
            gc.collect()
            torch.cuda.empty_cache()
    return max_adv_accuracy_iteration_list, max_adv_accuracy_noise_list, \
           max_adv_accuracy_list, max_adv_reg_accuracy_list


def init_model(args, noise, m_forward, add_args, device, dtype):
    smoothing_args = {'noise_sd': noise, 'm_forward': m_forward, 'smooth': args.smooth}
    if args.no_norm:
        smoothing_args['normalization'] = None
    else:
        smoothing_args['normalization'] = args.dataset_name
        if args.no_var_norm:
            smoothing_args['normalization'] = args.dataset_name + "_no_var"
    model = Smooth(args.net(**add_args), **smoothing_args)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    print("Number of parameters {}".format(num_parameters))

    # define loss function (criterion) and optimizer
    criterion = CrossEntropyLoss()

    model, criterion = model.to(device=device, dtype=dtype), criterion.to(device=device, dtype=dtype)
    return model, criterion


def init_hyper_params(args, add_args, noise, m_forward, logger, device, dtype, resume_path=None):
    model, criterion = init_model(args, noise, m_forward, add_args, device, dtype)

    train_loader, val_loader, _ = args.get_loaders(args.dataset, args.data, args.batch_size, args.batch_size,
                                                   args.workers)

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.decay, nesterov=args.nesterov_momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.decay)
    else:
        raise ValueError('Wrong optimzier!')

    # optionally resume from a checkpoint
    if resume_path is None:
        resume_path = args.resume
    if resume_path:
        model, found = resume_model(args, model, resume_path, optimizer, logger, device)
        if not found:
            logger.error("=> no checkpoint found at '{}'".format(resume_path))
            print("=> no checkpoint found at '{}'".format(resume_path))
            raise ValueError('No Checkpoint Found!')

    for layer in model.modules():
        from layers import NoisedConv2DColored
        if isinstance(layer, NoisedConv2DColored):
            try:
                logger.debug("Mean of alphas_diag_w is {}+-{} ({}) ".format(
                    torch.mean(torch.abs(layer.alphad_w)),
                    torch.std(torch.abs(layer.alphad_w)),
                    torch.max(torch.abs(layer.alphad_w))))
                logger.debug("Mean of alphas_factor_w is {}+-{} ({}) ".format(
                    torch.mean(torch.abs(layer.alphaf_w)),
                    torch.std(layer.alphaf_w),
                    torch.max(torch.abs(layer.alphaf_w))))
            except:
                pass
            try:
                logger.debug("Mean of alphas_diag_a is {}+-{} ({})  ".format(
                    torch.mean(torch.abs(layer.alphad_i)),
                    torch.std(torch.abs(layer.alphad_i)),
                    torch.max(torch.abs(layer.alphad_i))))
                logger.debug("Mean of alphas_factor_a is {}+-{} ({}) ".format(
                    torch.mean(torch.abs(layer.alphaf_i)),
                    torch.std(layer.alphaf_i),
                    torch.max(torch.abs(layer.alphaf_i))))
            except:
                pass

    # return model, val_loader, criterion
    return model, train_loader, criterion


def resume_model(args, model, resume_path, optimizer, logger, device):
    if os.path.isfile(resume_path):
        logger.info("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=device)
        args.start_epoch = checkpoint['epoch'] - 1
        # best_test = checkpoint['best_prec1']
        model.load_state_dict(transform_checkpoint(checkpoint['state_dict']))
        # optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    elif os.path.isdir(resume_path):
        checkpoint_path = os.path.join(resume_path, 'checkpoint.pth.tar')
        csv_path = os.path.join(resume_path, 'results.csv')
        logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        args.start_epoch = checkpoint['epoch'] - 1
        # best_test = checkpoint['best_prec1']
        model.load_state_dict(transform_checkpoint(checkpoint['state_dict']))
        # optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        data = []
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    else:
        return model, False
    return model, True


def run_attacks(args, model, val_loader, criterion, logger, device, dtype, att_objects=None):
    if att_objects is None:
        att_objects = []
        for i, att in enumerate(args.attacks):
            att_object = args.attack(model, criterion, **args.attacks_add_params[i])

            att_objects.append(att_object)

    test_loss_list, accuracy1_list, accuracy5_list, test_loss_a_list, \
    accuracy1_a_list, accuracy5_a_list, rad_list, pred_prob_list, pred_prob_var_list = [], [], [], [], [], [], [], [], []

    for i, att_object in enumerate(att_objects):
        if args.attacks_tar[i]:
            _, test_loss, accuracy1, accuracy5, test_loss_a, accuracy1_a, accuracy5_a, rad, pred_prob, pred_prob_var = \
                targeted_attack(model, val_loader, criterion, None, 0, args.experiment_name, logger, 0, att_object,
                                args.attacks_eps[i], args.num_classes, device, dtype,
                                calc_prob=not args.no_pred_prob)
        else:
            _, test_loss, accuracy1, accuracy5, test_loss_a, accuracy1_a, accuracy5_a, rad, pred_prob, pred_prob_var = \
                attack(model, val_loader, criterion, None, 0, args.experiment_name, logger, 0, att_object,
                       args.attacks_eps[i], device, dtype, calc_prob=not args.no_pred_prob)

            test_loss_list.append(test_loss)
            accuracy1_list.append(accuracy1)
            accuracy5_list.append(accuracy5)
            test_loss_a_list.append(test_loss_a)
            accuracy1_a_list.append(accuracy1_a)
            accuracy5_a_list.append(accuracy5_a)
            rad_list.append(rad)
            pred_prob_list.append(pred_prob)
            pred_prob_var_list.append(pred_prob_var)

    test_loss = np.mean(test_loss_list)
    accuracy1 = np.mean(accuracy1_list)
    accuracy5 = np.mean(accuracy5_list)
    cer_rad = np.mean(rad_list)
    pred_prob = np.mean(pred_prob_list)
    pred_prob_var = np.mean(pred_prob_var_list)

    return test_loss, accuracy1, accuracy5, cer_rad, pred_prob, pred_prob_var, test_loss_a_list, accuracy1_a_list, accuracy5_a_list


def main():
    args = get_args()
    device, dtype = args.device, args.dtype
    noise_sd = args.noise_sd
    m_forward = args.m_forward

    logger = get_logger(args, name='CPNI EPGD run_attack')
    if args.test_hyper_params:
        max_adv_accuracy_iteration_list, max_adv_accuracy_noise_list, \
        max_adv_accuracy_list, max_adv_reg_accuracy_list = test_hyper(args, logger)
        if not args.run_optimal:
            return
        noise_sd = max_adv_accuracy_noise_list[0]
        m_forward = max_adv_accuracy_iteration_list[0]

    add_args = {'weight_noise': args.weight_noise, 'act_noise_a': args.act_noise_a, 'act_noise_b': args.act_noise_b,
                'rank': args.noise_rank, 'noised_strength': args.weight_noise_d, 'noisef_strength': args.weight_noise_f,
                'num_classes': args.num_classes, 'width': args.width}

    model, val_loader, criterion = init_hyper_params(args, add_args, noise_sd, m_forward, logger, device, dtype)

    att_object = args.attack(model, criterion, **args.attacks_add_params[0])

    att_objects = None
    if not args.multi_att:
        att_objects = [att_object]

    _, accuracy1, accuracy5, _, _, _, _, accuracy1_a_list, accuracy5_a_list = run_attacks(args, model, val_loader, criterion,
                                                                                    logger, device, dtype, att_objects)
    print("accuracy1")
    print(accuracy1)
    print("accuracy1_a_list")
    print(accuracy1_a_list)
    print("accuracy5")
    print(accuracy5)
    print("accuracy5_a_list")
    print(accuracy5_a_list)


def transform_checkpoint(cp):
    new_cp = {}
    for entry in cp:
        new_name=entry.replace('module.', '')
        if new_name.startswith('1.'):
            new_name=new_name[2:]
        new_cp[new_name] = cp[entry]
    return new_cp


if __name__ == '__main__':
    main()
