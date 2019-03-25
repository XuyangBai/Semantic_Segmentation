import random
import numpy as np
import torch
import numbers
from PIL import Image


class Scale(object):
    def __init__(self):
        pass

    def __call__(self, img, label):
        return img, label


def flip_tensor(tensor, axis):
    if len(tensor.size()) == 1:
        return tensor
    tNp = np.flip(tensor.numpy(), axis).copy()
    return torch.from_numpy(tNp)


class RandomFlip2d(object):
    def __init__(self, axis_switch=(1, 1)):
        self.axis_switch = axis_switch

    def __call__(self, img):
        if self.axis_switch[0]:
            if random.randint(0, 1) == 1:
                img = flip_tensor(img, -2)
        if self.axis_switch[1]:
            if random.randint(0, 1) == 1:
                img = flip_tensor(img, -1)
        return img


def crop_size_correct(sp, ep, this_size):
    assert ep - sp <= this_size, 'Invalid crop size.'
    if sp < 0:
        ep -= sp
        sp -= sp
    elif ep > this_size:
        sp -= (ep - this_size)
        ep -= (ep - this_size)

    return sp, ep


def crop(tensor, locations):
    """
    ''location'' is a tuple indicating locations of start and end points
    """
    s = tensor.size()
    if len(locations) == 4:
        x1, y1, x2, y2 = locations
        x1, x2 = crop_size_correct(x1, x2, s[-1])
        y1, y2 = crop_size_correct(y1, y2, s[-2])
        return tensor[..., y1:y2, x1:x2]
    else:
        raise RuntimeError('Invalid crop size dimension.')


class RandomCrop2d_cls(object):
    """Crops the given (img, label) at a random location to have a region of
    the given size. size can be a tuple (target_depth, target_height, target_width)
    or an integer, in which case the target will be of a cubic shape (size, size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.size()[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        loc = (x1, y1, x1 + tw, y1 + th)
        return crop(img, loc)
