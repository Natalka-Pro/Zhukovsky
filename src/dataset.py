from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from .functions import seed_everything


class My_Dataset(Dataset):
    def __init__(
        self, kind, images_dir, augmentation, transform, threshold, deterministic=True, seed=42
    ):
        self.kind = 1 if kind == "pos" else 0
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

            i = 0
            while len(self.X) < self.required_len:
                image = Image.open(self.image_paths[i % self.real_len])
                i += 1

                dol = 1
                k = 0
                while dol >= threshold:
                    transformed_image = self.transform(image)

                    x, col = np.unique(transformed_image.max(dim = 0)[0], return_counts = True)
                    ones = col[np.where(x == 1)[0]][0]
                    dol = ones / col.sum() # доля белого
                    k += 1
                    if k > 100:
                        print(f"Доля белого больше у {i} картинки!!!")
                        break

                self.X.append((transformed_image, self.kind))

    def __len__(self):
        return self.required_len

    def __getitem__(self, idx):
        if idx < len(self):

            if self.deterministic:
                return self.X[idx]
            else:
                image = Image.open(self.image_paths[idx % self.real_len])
                return self.transform(image), self.kind
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
