import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .functions import seed_everything


class My_Dataset(Dataset):
    def __init__(
        self,
        kind,
        images_dir,
        augmentation,
        transform,
        threshold=False,
        seed=42,
        deterministic=True,
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

                if threshold:
                    dol = 1
                    k = 0
                    while dol >= threshold:
                        transformed_image = self.transform(image)

                        x, col = np.unique(
                            transformed_image.max(dim=0)[0], return_counts=True
                        )
                        ones = col[np.where(x == 1)[0]][0]
                        dol = ones / col.sum()  # доля белого
                        k += 1
                        if k > 100:
                            print(f"Доля белого больше у {i} картинки!!!")
                            print(self.image_paths[i % self.real_len])
                            break
                else:
                    transformed_image = self.transform(image)

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


class TripletDataset(Dataset):
    def __init__(self, dataset1, dataset2, required_len, deterministic=True, seed=42):
        self.d1 = dataset1
        self.d2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        self.required_len = required_len
        self.deterministic = deterministic

        if self.deterministic:
            seed_everything(seed)
            self.X = []

            for i in range(self.required_len):
                anchor, positive = random.sample(range(self.len1), 2)
                negative = random.randrange(self.len2)

                ds = [self.d1, self.d1, self.d2]
                idxs = [anchor, positive, negative]
                self.X.append(tuple(dataset[i][0] for dataset, i in zip(ds, idxs)))

    def __len__(self):
        return self.required_len

    def __getitem__(self, idx):
        if idx < len(self):

            if self.deterministic:
                return self.X[idx]
            else:
                anchor, positive = random.sample(range(self.len1), 2)
                negative = random.randrange(self.len2)

                ds = [self.d1, self.d1, self.d2]
                idxs = [anchor, positive, negative]
                return tuple(dataset[i][0] for dataset, i in zip(ds, idxs))
        else:
            raise IndexError


class Emb_Dataset(Dataset):
    def __init__(self, model, dataset, device):
        self.dataset = dataset
        self.model = model.eval()
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx < len(self):
            img, cls = self.dataset[idx]
            emb = self.model(img[None, :].to(self.device))[0]
            return emb, cls
        else:
            raise IndexError
