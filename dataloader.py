from dataset import OutdoorDataset
import torchvision.transforms as transforms
import transforms as mytfm
from torch.utils.data import DataLoader


def get_data_loader(root, batch_size, split='train', num_workers=4):
    # TODO: implement the data augmentation.
    preprocess = transforms.Compose([

        transforms.ToTensor(),
        # transforms.Normalize(mean=, std=)
        mytfm.RandomFlip2d((1, 1)),
        mytfm.RandomCrop2d((256, 256)),
    ])
    dset = OutdoorDataset(root=root, split=split, transform=preprocess)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader
