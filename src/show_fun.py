import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from .classifier import get_predictions


def open_image(image):
    # чтобы изображение не поворачивалось на 90 градусов, если w < h
    # https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
    img = Image.open(image)
    img = ImageOps.exif_transpose(img)
    return img


def graph2(x, y1, y2, title="", ylog=True, ylabel=""):

    plt.title(title, fontsize=13)

    plt.plot(x, y1, color="g", label="Train")
    plt.plot(x, y2, color="m", label="Val")

    plt.grid(True)
    plt.ylabel(ylabel, fontsize=10)
    plt.xlabel("Номер эпохи", fontsize=10)

    if ylog:
        plt.yscale("log")

    plt.legend(fontsize=10)


def graph_logs(logs, num=None, ylog=True):
    plt.figure(figsize=(15, 6))

    if not num:
        num = len(logs["epoch"])

    x = logs["epoch"][:num]
    y1 = logs["train_loss"][:num]
    y2 = logs["val_loss"][:num]

    plt.subplot(1, 2, 1)
    graph2(x, y1, y2, title="", ylog=ylog, ylabel="loss")

    x = logs["epoch"][:num]
    y1 = logs["train_accuracy"][:num]
    y2 = logs["val_accuracy"][:num]

    plt.subplot(1, 2, 2)
    graph2(x, y1, y2, title="", ylog=ylog, ylabel="accuracy")

    plt.show()


def plot_transformed_images(image_paths, transform, n=3):
    # https://www.learnpytorch.io/04_pytorch_custom_datasets/
    # random_image_paths = random.sample(image_paths, k=n)

    if n:
        image_paths = image_paths[:n]

    for image_path in image_paths:

        f = Image.open(image_path)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(f)
        plt.title(f"Original \nSize: {f.size}")
        plt.axis("off")
        # Transform and plot image
        # Note: permute() will change shape of image to suit matplotlib
        # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
        transformed_image = transform(f).permute(1, 2, 0)
        plt.subplot(1, 2, 2)
        plt.imshow(transformed_image)
        plt.title(f"Transformed \nSize: {tuple(transformed_image.shape)}")
        plt.axis("off")
        plt.show()


# def show_batch(images, labels, n=4):
#     f, axes = plt.subplots(n // 4, 4, figsize=(30, 10))

#     for i, axis in enumerate(axes):
#         # переводим картинку из тензора в numpy
#         img = images[i].numpy()
#         # переводим картинку в размерность (длина, ширина, цветовые каналы)
#         img = np.transpose(img, (1, 2, 0))

#         axes[i].imshow(img)
#         axes[i].set_title(labels[i].numpy())

#     plt.show()


def show_images(images, labels, n=4):
    num_pic = min(len(images), n)
    width, height = 4, num_pic // 4 + 1

    for i in range(num_pic):
        if i % width == 0:
            plt.figure(figsize=(6.4 * width, 6))

        plt.subplot(1, width, i % width + 1)

        img = images[i]
        img = np.transpose(img, (1, 2, 0))

        plt.imshow(img)
        # if torch.is_tensor(labels[i]):
        #     title = labels[i].numpy()
        # else:
        #     title = labels[i]
        # x, col = np.unique(img.max(dim=2)[0], return_counts=True)
        # ones = col[np.where(x == 1)[0]][0]
        # dol = ones / col.sum()

        number = labels[i]
        # number = "    ".join([str(_) for _ in labels[i]])
        title = (
            f"№ {int(number[0])}:   {round(number[1], 5)}"  # , доля белого - {dol:.4f}"
        )

        plt.title(title)

        if i % width == width - 1:
            plt.tight_layout()
            plt.show()

    plt.show()


def show_voting():
    pass


def show_result(
    model,
    dataset,
    threshold,
    batch_size,
    device,
    greater=True,
    col=None,
    sort=False,
):

    X, y_true, y_pred, y_prob = get_predictions(model, dataset, batch_size, device)

    if sort:
        if greater:
            _, indices = torch.sort(y_prob, descending=True)
        else:
            _, indices = torch.sort(y_prob, descending=False)
        indices = indices.numpy()
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        y_prob = y_prob[indices]
        X = X[indices]
    else:
        indices = np.arange(len(X))

    if greater:
        idx = np.where(y_prob > threshold)[0]
    else:
        idx = np.where(y_prob < threshold)[0]

    sign = ">" if greater else "<"
    print(f"prob {sign} {threshold}\ncount : {len(idx)} out of {len(X)}")
    if col:
        idx = idx[:col]
    print(f"pic idx : {list(indices[idx])}")

    labels = np.stack((indices[idx], y_prob[idx]), axis=1)  # n x 2

    show_images(X[idx], labels, n=len(idx))

    return X, y_true, y_pred, y_prob, indices
