from torch.utils.data import Dataset
import os
from PIL import Image
from .functions import seed_everything


class Positive_Dataset(Dataset):
    def __init__(
        self, images_dir, augmentation, transform, deterministic=True, seed=42
    ):
        self.augmentation = augmentation
        self.transform = transform
        self.deterministic = deterministic

        self.files_names = sorted(os.listdir(images_dir))
        self.image_paths = [os.path.join(images_dir, i) for i in self.files_names]

        self.real_len = len(self.files_names)
        self.required_len = self.augmentation * self.real_len

        if self.deterministic:
            seed_everything(seed)
            self.X = []

            for i in range(self.required_len):
                image = Image.open(self.image_paths[i % self.real_len])
                self.X.append((self.transform(image), 1))

    def __len__(self):
        return self.required_len

    def __getitem__(self, idx):
        if idx < len(self):

            if self.deterministic:
                return self.X[idx]
            else:
                image = Image.open(self.image_paths[idx % self.real_len])
                return self.transform(image), 1
        else:
            raise IndexError


class Negative_Dataset(Dataset):
    def __init__(
        self, images_dir, augmentation, transform, deterministic=True, seed=42
    ):
        self.augmentation = augmentation
        self.transform = transform
        self.deterministic = deterministic

        self.files_names = sorted(os.listdir(images_dir))
        self.image_paths = [os.path.join(images_dir, i) for i in self.files_names]

        self.real_len = len(self.files_names)
        self.required_len = self.augmentation * self.real_len

        if self.deterministic:
            seed_everything(seed)
            self.X = []

            for i in range(self.required_len):
                image = Image.open(self.image_paths[i % self.real_len])
                self.X.append((self.transform(image), 0))

    def __len__(self):
        return self.required_len

    def __getitem__(self, idx):
        if idx < len(self):

            if self.deterministic:
                return self.X[idx]
            else:
                image = Image.open(self.image_paths[idx % self.real_len])
                return self.transform(image), 0
        else:
            raise IndexError


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
