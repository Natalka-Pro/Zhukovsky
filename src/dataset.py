from torch.utils.data import Dataset
import os
from PIL import Image


class Positive_Dataset(Dataset):
    def __init__(self, images_dir, augmentation, transform):
        self.transform = transform
        self.augmentation = augmentation

        files_names = sorted(os.listdir(images_dir))
        self.image_paths = [os.path.join(images_dir, i) for i in files_names]

        self.real_len = len(files_names)

    def __len__(self):
        return self.augmentation * len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx % self.real_len])
        return self.transform(image), 1


class Negative_Dataset(Dataset):
    def __init__(self, images_dir, augmentation, transform):
        self.transform = transform
        self.augmentation = augmentation

        files_names = sorted(os.listdir(images_dir))
        self.image_paths = [os.path.join(images_dir, i) for i in files_names]

        self.real_len = len(files_names)

    def __len__(self):
        return self.augmentation * len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx % self.real_len])
        return self.transform(image), 0


def pos_neg(dataloader):
    num_pos = num_neg = num = 0

    for batch in dataloader:
        images, labels = batch

        col = images.size(0)
        pos = labels.sum()

        num += col
        num_pos += pos
        num_neg += col - pos

    return {"1": num_pos.item(), "0": num_neg.item(), "total": num}
