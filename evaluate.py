def calculate_loss(predict, true):
    # TODO: implement the evaluation (IOU, accuracy, loss)
    res = {
        'loss': 0.1,
        'acc': 0.2, # or acc for every category,
        'iou': 0.3
    }
    return res


def evaluate(model, loader, gpu_mode):
    loss_buf = []
    for i_batch, (img, mask) in enumerate(loader):
        if gpu_mode:
            img = img.cuda()
            mask = mask.cuda()
        output = model(img)
        # calculate the loss and acc for this batch
        res = calculate_loss(output, mask)
        loss_buf.append(res['loss'].detach().cpu().numpy())

    return res