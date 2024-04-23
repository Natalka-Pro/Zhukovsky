import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 900_000_000
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix as conf_matrix
import seaborn as sns


def open_image(image):
    # чтобы изображение не поворачивалось на 90 градусов, если w < h
    # https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
    img = Image.open(image)
    img = ImageOps.exif_transpose(img)
    return img


def seed_everything(seed: int):
    # import random, os
    # import numpy as np
    # import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def plot_transformed_images(image_paths, transform, n=3):
    # https://www.learnpytorch.io/04_pytorch_custom_datasets/
    random_image_paths = random.sample(image_paths, k=n)

    for image_path in random_image_paths:

        f = Image.open(image_path)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(f)
        ax[0].set_title(f"Original \nSize: {f.size}")
        # ax[0].axis("off")

        # Transform and plot image
        # Note: permute() will change shape of image to suit matplotlib
        # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
        transformed_image = transform(f).permute(1, 2, 0)
        ax[1].imshow(transformed_image)
        ax[1].set_title(f"Transformed \nSize: {transformed_image.size()}")
        # ax[1].axis("off")


def show_images(images, labels, n=4):
    f, axes = plt.subplots(n // 4, 4, figsize=(30, 10))

    for i, axis in enumerate(axes):
        # переводим картинку из тензора в numpy
        img = images[i].numpy()
        # переводим картинку в размерность (длина, ширина, цветовые каналы)
        img = np.transpose(img, (1, 2, 0))

        axes[i].imshow(img)
        axes[i].set_title(labels[i].numpy())

    plt.show()


def confusion_matrix(y_true, y_pred):
    cm = conf_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt="g", xticklabels=["0", "1"], yticklabels=["0", "1"])

    plt.xlabel("Prediction", fontsize=13)
    plt.ylabel("True", fontsize=13)
    plt.title("Confusion Matrix", fontsize=17)
    plt.show()
    return cm


# ВАШ КОД: постройте и обучите нейросеть
# model.children() выдает список сабмодулей нейросети
# в нашем случае это блоки resnet
def create_model(model, num_freeze_layers, num_out_classes):
    # замена последнего слоя сети
    model.fc = nn.Linear(512, num_out_classes)

    # заморозка слоев
    for i, layer in enumerate(model.children()):
        if i < num_freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False

    return model