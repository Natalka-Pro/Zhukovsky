import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 900_000_000
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix as conf_matrix


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def path_join(*l):
    return os.path.join(*l)


def dir_paths(dir):
    paths = []
    for p in sorted(os.listdir(dir)):
        path = os.path.join(dir, p)
        paths.append(path)
    return paths


def path_split(path):
    p = re.split(r"/", path)
    return p


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def confusion_matrix(y_true, y_pred):
    cm = conf_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt="g", xticklabels=["0", "1"], yticklabels=["0", "1"])

    plt.xlabel("Prediction", fontsize=13)
    plt.ylabel("True", fontsize=13)
    plt.title("Confusion Matrix", fontsize=17)
    plt.show()
    return cm


def distribution(y_prob):
    plt.hist(y_prob.numpy(), bins=50)
    plt.xlim((0, 1))


# ВАШ КОД: постройте и обучите нейросеть
# model.children() выдает список сабмодулей нейросети
# в нашем случае это блоки resnet
# def create_model(model, num_freeze_layers, num_out_classes):
#     # замена последнего слоя сети
#     model.fc = nn.Linear(512, num_out_classes)

#     # заморозка слоев
#     for i, layer in enumerate(model.children()):
#         if i < num_freeze_layers:
#             for param in layer.parameters():
#                 param.requires_grad = False

#     return model


def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# for i, layer in enumerate(model.children()):
#     print(f"{i} layer")
#     # print(layer)
#     for param in layer.parameters():
#         print(f"  grad = {param.requires_grad}, {param.shape}, {param.numel()}")


def create_model(model, num_non_freeze, num_out_classes, verbose=False):
    # замена последнего слоя сети
    model.fc = nn.Linear(512, num_out_classes)

    num_param = number_of_parameters(model)
    num_freeze = num_param - num_non_freeze

    # заморозка слоев
    cur_freeze = 0
    for i, layer in enumerate(model.children()):
        for param in layer.parameters():
            if param.requires_grad:
                if cur_freeze >= num_freeze:
                    return model

                param.requires_grad = False
                cur_freeze += param.numel()
                if verbose:
                    print(num_param - cur_freeze)
