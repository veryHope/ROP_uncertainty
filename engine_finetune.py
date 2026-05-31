# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import csv
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
import time
from typing import Iterable
import torch
import torch.nn.functional as F
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import scipy.stats


def set_dropout_p(model, p=0.1, last_n_blocks=3):
    blocks = model.blocks3
    if last_n_blocks is not None:
        blocks = blocks[-last_n_blocks:]

    for block in blocks:
        if hasattr(block, "mlp"):
            if hasattr(block.mlp, "drop"):
                dropout_layer = block.mlp.drop
                if isinstance(dropout_layer, torch.nn.Dropout):
                    dropout_layer.p = p
                    dropout_layer.train()
                    print(f"Enable dropout: p={p}")


def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)
    return (A + B)


def compute_evidence_and_uncertainty(pred, num_classes):
    evidence = F.softplus(pred)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    b = E / S
    expected_prob = F.softmax(b, dim=1)
    uncertainty = num_classes / S
    return expected_prob, uncertainty


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if getattr(args, 'use_edl', False):
                evidence = F.softplus(outputs)
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                E = alpha - 1
                b = E / (S.expand(E.shape))
                Tem_Coef = epoch * (0.5 / args.epochs) + 0.5
                loss_CE = criterion(b / Tem_Coef, targets)
                loss_EDL = ce_loss(targets, alpha, args.nb_classes, epoch, args.epochs, device)
                loss_ACE = torch.mean(loss_EDL)
                loss = loss_CE + loss_ACE
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        if data_iter_step % 50 == 0:
            time.sleep(0.5)

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def compute_auc_ci_delong(y_true, y_prob, alpha=0.95):

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
   
    pos_prob = y_prob[y_true == 1]
    neg_prob = y_prob[y_true == 0]

    m = len(pos_prob)
    n = len(neg_prob)

    theta = 0.0
    for p in pos_prob:
        theta += np.sum(p > neg_prob) + 0.5 * np.sum(p == neg_prob)
    auc = theta / (m * n)

    V10 = np.zeros(m)
    for i in range(m):
        V10[i] = (np.sum(pos_prob[i] > neg_prob) + 0.5 * np.sum(pos_prob[i] == neg_prob)) / n

    V01 = np.zeros(n)
    for j in range(n):
        V01[j] = (np.sum(neg_prob[j] < pos_prob) + 0.5 * np.sum(neg_prob[j] == pos_prob)) / m

    S10 = np.var(V10, ddof=1)
    S01 = np.var(V01, ddof=1)
    var_auc = (S10 / m) + (S01 / n)

    z_score = scipy.stats.norm.ppf((1 + alpha) / 2)
    margin_of_error = z_score * np.sqrt(var_auc)

    lower_bound = np.clip(auc - margin_of_error, 0.0, 1.0)
    upper_bound = np.clip(auc + margin_of_error, 0.0, 1.0)

    formatted_string = f"{auc:.4f} ({lower_bound:.4f}-{upper_bound:.4f})"

    return auc, lower_bound, upper_bound, formatted_string


@torch.no_grad()
def evaluate(data_loader, model, device, save_path='', args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    list_prob_all = []
    list_lb_all = []
    list_path_all = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        if len(batch) == 3:
            images, target, paths = batch[0], batch[1], batch[2]
        else:
            images, target = batch[0], batch[-1]
            paths = [''] * images.shape[0]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        list_prob = torch.softmax(output, dim=1).cpu().numpy()[:, 1].tolist()
        list_lb = target.cpu().numpy().tolist()
        list_lb_all = list_lb_all + list_lb
        list_prob_all = list_prob_all + list_prob
        list_path_all = list_path_all + list(paths)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)


    np_lb_all = np.array(list_lb_all, dtype=np.int32).flatten()
    np_prob_all = np.array(list_prob_all, dtype=np.float32).flatten()
    np_path_all = np.array(list_path_all, dtype=np.str_).flatten()

    ap = average_precision_score(np_lb_all, np_prob_all)
    auc, ci_lower, ci_upper, auc_ci_str = compute_auc_ci_delong(np_lb_all, np_prob_all)
    print(auc, ci_lower, ci_upper, auc_ci_str)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['path', 'labels', 'prob1'])
            for i in range(len(np_lb_all)):
                writer.writerow([np_path_all[i], np_lb_all[i], np_prob_all[i]])
        print(f"Results saved to: {save_path}")

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} AUC {auc_ci} AP {ap:.4f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, auc=auc, ap=ap, auc_ci=auc_ci_str))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc


@torch.no_grad()
def evaluate_edl(data_loader, model, device, num_classes=2, save_path=''):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    list_prob_all = []
    list_lb_all = []
    list_uncertainty_all = []
    list_path_all = []

    print("Using Evidential Deep Learning for uncertainty estimation")
    print(len(data_loader))

    for batch in metric_logger.log_every(data_loader, 10, header):
        if len(batch) == 3:
            images, target, paths = batch[0], batch[1], batch[2]
        else:
            images, target = batch[0], batch[-1]
            paths = [''] * images.shape[0]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        expected_prob, uncertainty = compute_evidence_and_uncertainty(output, num_classes)

        prob = expected_prob.cpu().numpy()
        uncertainty_arr = uncertainty.cpu().numpy()

        list_prob = prob[:, 1].tolist()
        list_uncertainty = uncertainty_arr[:, 0].tolist()
        list_lb = target.cpu().numpy().tolist()

        list_lb_all = list_lb_all + list_lb
        list_prob_all = list_prob_all + list_prob
        list_uncertainty_all = list_uncertainty_all + list_uncertainty
        list_path_all = list_path_all + list(paths)

        batch_size = images.shape[0]

        acc1, _ = accuracy(torch.from_numpy(prob).to(device), target, topk=(1, 2))
        if not isinstance(acc1, torch.Tensor):
            acc1 = torch.tensor(acc1, device=device)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    np_lb_all = np.array(list_lb_all, dtype=np.int32).flatten()
    np_prob_all = np.array(list_prob_all, dtype=np.float32).flatten()
    np_uncertainty_all = np.array(list_uncertainty_all, dtype=np.float32).flatten()
    np_path_all = np.array(list_path_all, dtype=np.str_).flatten()

    auc = roc_auc_score(np_lb_all, np_prob_all)
    ap = average_precision_score(np_lb_all, np_prob_all)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['path', 'labels', 'prob1', 'uncertainty'])
            for i in range(len(np_lb_all)):
                writer.writerow([np_path_all[i], np_lb_all[i], np_prob_all[i], np_uncertainty_all[i]])
        print(f"EDL results saved to: {save_path}")

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} AUC {auc:.4f} AP {ap:.4f}'
          .format(top1=metric_logger.acc1, auc=auc, ap=ap))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc


@torch.no_grad()
def evaluate_mc_dropout(data_loader, model, device, n_samples=20, save_path=''):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.train()

    list_prob_all = []
    list_lb_all = []
    list_var_all = []
    list_path_all = []

    print(f"MC Dropout: {n_samples} samples")
    print(len(data_loader))

    for batch in metric_logger.log_every(data_loader, 10, header):
        if len(batch) == 3:
            images, target, paths = batch[0], batch[1], batch[2]
        else:
            images, target = batch[0], batch[-1]
            paths = [''] * images.shape[0]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        list_output = []
        list_loss = []

        for i in range(n_samples):
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
                output = torch.softmax(output, dim=1)
                list_output.append(output)
                list_loss.append(loss)

        stacked_output = torch.stack(list_output)
        output = torch.mean(stacked_output, dim=0)
        var = torch.var(stacked_output, dim=0, unbiased=True)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        list_prob = output.cpu().numpy()[:, 1].tolist()
        list_var = var.cpu().numpy()[:, 1].tolist()
        list_lb = target.cpu().numpy().tolist()

        list_lb_all = list_lb_all + list_lb
        list_prob_all = list_prob_all + list_prob
        list_var_all = list_var_all + list_var
        list_path_all = list_path_all + list(paths)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    np_lb_all = np.array(list_lb_all, dtype=np.int32).flatten()
    np_prob_all = np.array(list_prob_all, dtype=np.float32).flatten()
    np_var_all = np.array(list_var_all, dtype=np.float32).flatten()
    np_path_all = np.array(list_path_all, dtype=np.str_).flatten()

    auc = roc_auc_score(np_lb_all, np_prob_all)
    ap = average_precision_score(np_lb_all, np_prob_all)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['path', 'labels', 'prob1', 'uncertainty'])
            for i in range(len(np_lb_all)):
                writer.writerow([np_path_all[i], np_lb_all[i], np_prob_all[i], np_var_all[i]])
        print(f"MC Dropout results saved to: {save_path}")

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} AUC {auc:.4f} AP {ap:.4f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, auc=auc, ap=ap))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc