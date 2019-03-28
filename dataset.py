import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import random


class OutdoorDataset(data.Dataset):
    def __init__(self, root, split='train'):
        self.dataset = 'OutdoorDataset'
        self.root = root  # 把解压后的train文件夹放在dataset目录下
        self.split = split
        self.datapath = []  # every element contains path for image & path for mask
        with open(os.path.join(self.root, split), 'r') as f:
            ids = f.readlines()
        for id in ids:
            self.datapath.append(
                {
                    'name': id,
                    'img': os.path.join(self.root, split + '/images/' + id + '.png'),
                    'mask': os.path.join(self.root, split + '/labels/' + id + '.png')
                }
            )

    def transform(self, image, mask):
        # Resize
        resize_img = transforms.Resize(size=(960, 720), interpolation=Image.BILINEAR)
        resize_mask = transforms.Resize(size=(960, 720), interpolation=Image.NEAREST)
        image = resize_img(image)
        mask = resize_mask(mask)

        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(800, 600))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

    def __getitem__(self, index):
        img_path = self.datapath[index]['img']
        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32)
        mask_path = self.datapath[index]['mask']
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int8)

        img, mask = self.transform(img, mask)

        return img, mask

    def __len__(self):
        return len(self.datapath)
