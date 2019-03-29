# System libs
import time
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
# Local libs
from utils import AverageMeter, accuracy, intersectionAndUnion


# TODO: implement the evaluation (IOU, accuracy, loss)
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
    time_meter = AverageMeter()

    # model mode
    model.eval()

    for i_batch, (img, mask) in enumerate(loader):
        if gpu_mode:
            img = img.cuda()
            mask = mask.cuda()

        torch.cuda.synchronize()
        tic = time.perf_counter()
        output = model(img)
        torch.cuda.synchronize()

        time_meter.update(time.perf_counter() - tic)

        # calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, mask)
        loss_value = loss.data.cpu().numpy()
        loss_meter.update(loss_value)

        # calculate accuracy
        acc = accuracy(output, mask)
        acc_meter.update(acc)

        # calculate iou
        intersection, union = intersectionAndUnion(output, mask, num_class)
        inter_meter.update(intersection)
        union_meter.update(union)

    # summary
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    iou_mean = iou.mean().cpu().numpy()
    acc_mean = acc_meter.average().cpu().numpy()
    loss_mean = loss_meter.average().cpu().numpy()

    res['loss'] = loss_mean
    res['acc'] = acc_mean
    res['iou'] = iou_mean

    return res
