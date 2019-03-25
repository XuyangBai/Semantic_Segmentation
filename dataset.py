import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np


class OutdoorDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.dataset = 'OutdoorDataset'
        self.root = root  # 把解压后的train文件夹放在dataset目录下
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.datapath = []  # every element contains path for image & path for mask
        for file in os.listdir(""):
            self.datapath.append(
                {
                    'name': file,
                    'img': file,
                    'mask': file
                }
            )

    def __getitem__(self, index):
        img_path = self.datapath[index]['img']
        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32)
        mask_path = self.datapath[index]['mask']
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int8)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.datapath)
