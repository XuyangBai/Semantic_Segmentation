import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import os
from matplotlib import pyplot as plt
import numpy as np
import random


class OutdoorDataset(data.Dataset):
    def __init__(self, root, split='train'):
        self.dataset = 'OutdoorDataset'
        self.root = root  # 把解压后的train文件夹放在dataset目录下
        self.split = split
        self.datapath = []  # every element contains path for image & path for mask
        with open(self.root + self.split + '/train.txt', 'r') as f:
            ids = f.readlines()
        for id in ids:
            id = id.replace("\n", "")
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
        mask_path = self.datapath[index]['mask']
        mask = Image.open(mask_path)

        img, mask = self.transform(img, mask)

        return img, mask

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    dataset = OutdoorDataset('data/')
    img, msk = dataset[0]
    img_np = img.numpy()
    img_np = np.transpose(img_np, [1, 2, 0])
    plt.imshow(img_np)
    plt.show()


    print(img.shape)
    print(msk.shape)
    msk_np = msk.numpy()
    msk_np = np.repeat(msk_np, 3, axis=0)
    print(msk_np.shape)
    msk_np = msk_np * 256 / 6
    msk_np = np.transpose(msk_np, [1, 2, 0])
    plt.imshow(msk_np)
    plt.show()
