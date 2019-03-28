from dataset import OutdoorDataset
import torchvision.transforms as transforms
import transforms as mytfm
from torch.utils.data import DataLoader


def get_data_loader(root, batch_size, split='train', num_workers=4):
    # TODO: implement the data augmentation.
    preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((960, 720)),
        transforms.RandomCrop((800, 600)),
        # transforms.RandomRotation((-20, 20)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=, std=)
    ])
    dset = OutdoorDataset(root=root, split=split, transform=preprocess)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader
