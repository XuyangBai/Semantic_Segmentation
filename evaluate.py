import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter, accuracy, intersectionAndUnion

def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

def cross_entropy2d(output, truth, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = output.size()
    log_p = F.log_softmax(output, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[truth.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = truth >= 0
    target = truth[mask]
    loss = F.nll_loss(log_p, target.long(), weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum().float()
    return loss


def evaluate(model, loader, gpu_mode, num_class=7):
    # output format
    res = {
        'loss': 0.1,
        'acc': 0.2,  # or acc for every category,
        'iou': 0.3
    }

    # metric meters
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    for i_batch, (img, mask) in enumerate(loader):
        if gpu_mode:
            img = img.cuda()
            mask = mask.cuda()

        output = model(img)
        # calculate loss
        loss = cross_entropy2d(output, mask)
        loss_value = loss.data.cpu().numpy()
        loss_meter.update(loss_value)

        output = output.max(1)[1]
        # calculate accuracy
        acc = accuracy(output, mask)
        acc_meter.update(acc)

        if gpu_mode:
            output = output.int().cpu().detach().numpy()
            mask = mask.int().cpu().detach().numpy()
        seg_pred = np.array(output)
        seg_gt = np.array(mask)
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, 7)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        
        # calculate iou
        # intersection, union = intersectionAndUnion(output, mask, num_class)
        # inter_meter.update(intersection)
        # union_meter.update(union)

    # summary
    iou = IU_array
    iou_mean = IU_array.mean()
    # iou = inter_meter.sum / (union_meter.sum + 1e-10)
    # iou_mean = iou.mean()
    acc_mean = acc_meter.average()
    loss_mean = loss_meter.average()

    res['loss'] = loss_mean
    res['acc'] = acc_mean
    res['iou'] = iou
    res['iou_mean'] = iou_mean

    return res
