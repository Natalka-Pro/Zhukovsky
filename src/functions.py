import os
import random
import numpy as np
import matplotlib.pyplot as plt
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 900_000_000
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix as conf_matrix
import seaborn as sns
from torch.utils.data import DataLoader


def get_predictions(model, dataset, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    pred_labels = []
    prob_labels = []
    true_labels = []
    X = []

    for i, batch in enumerate(dataloader):

        # так получаем текущий батч
        X_batch, y_batch = batch
        true_labels.append(y_batch)
        X.append(X_batch)

        with torch.no_grad():
            logits = model(X_batch.to(device))
            y_pred = torch.argmax(logits, dim=1)
            y_prob = torch.softmax(logits, dim=1)[:, 1]

            pred_labels.append(y_pred)
            prob_labels.append(y_prob)

    pred_labels = torch.cat(pred_labels)
    prob_labels = torch.cat(prob_labels)
    true_labels = torch.cat(true_labels)
    X = torch.cat(X)

    # print("pred_labels, true_labels, prob_labels, X")
    return pred_labels.cpu(), true_labels, prob_labels.cpu(), X


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


def create_model(model, num_non_freeze, num_out_classes):
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
                # print(num_param - cur_freeze)
