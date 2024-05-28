import json
import os
from time import gmtime, strftime, time

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from . import classifier, siamese
from .dataset import Emb_Dataset, My_Dataset, TripletDataset
from .dataset_fun import split_dataset
from .functions import create_model, number_of_parameters, seed_everything
from .train import load_logs, load_model, train


def hydra_config():
    initialize(config_path=".", version_base="1.3")
    config = compose(config_name="config.yaml")

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    return config


def config_model(CONF, model, kind):
    class Info:
        pass

    conf = Info()

    if kind == "siam":
        conf.train_epoch = siamese.train_epoch
        conf.eval_epoch = siamese.eval_epoch
        conf.loss_fn = nn.TripletMarginLoss(margin=CONF.siamese.margin, p=2)
        conf.optimizer = torch.optim.Adam(
            model.parameters(), lr=CONF.siamese.learning_rate
        )
        conf.n_epochs = CONF.siamese.n_epochs
        conf.path_log = CONF.siamese.path_log
        conf.path_model = CONF.siamese.path_model

    elif kind == "cl":
        conf.train_epoch = classifier.train_epoch
        conf.eval_epoch = classifier.eval_epoch
        conf.loss_fn = torch.nn.CrossEntropyLoss()
        conf.optimizer = torch.optim.Adam(
            model.parameters(), lr=CONF.classifier.learning_rate
        )
        conf.n_epochs = CONF.classifier.n_epochs
        conf.path_log = CONF.classifier.path_log
        conf.path_model = CONF.classifier.path_model

    conf.path_model = os.path.join(CONF.save_path, conf.path_model)
    if not os.path.exists(conf.path_model):
        os.makedirs(conf.path_model)

    conf.path_log = os.path.join(CONF.save_path, conf.path_log)

    return conf


def show_CONF(CONF, indent=8):
    d = OmegaConf.to_container(CONF)  # dict
    print(json.dumps(d, indent=indent))


def common_train(CONF, model, train_dataset, test_dataset, kind):
    train_loader = DataLoader(
        train_dataset, batch_size=CONF.loader.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONF.loader.batch_size, shuffle=False
    )

    seed_everything(CONF.seed)

    conf = config_model(CONF, model, kind)

    train(
        model,
        train_loader,
        test_loader,
        conf.train_epoch,
        conf.eval_epoch,
        conf.loss_fn,
        conf.optimizer,
        conf.n_epochs,
        CONF.device,
        kind,
        conf.path_log,
        conf.path_model,
    )


def load_best_model(CONF, model, train_dataset, test_dataset, kind):
    print("load_best_model:")
    conf = config_model(CONF, model, kind)

    log = f"# {{}} Epoch {{:{len(str(conf.n_epochs))}}} "
    log += f"train/val: loss {{:6.5f}}/{{:6.5f}}, acc:{{:7.3f}}%/{{:7.3f}}%"

    logs = load_logs(conf.path_log)
    idx = np.array(logs["val_accuracy"]).argmax()
    best_epoch = logs["epoch"][idx]

    params = (
        strftime("%Y-%m-%d %H:%M:%S", gmtime(time())),
        best_epoch,
        logs["train_loss"][idx],
        logs["val_loss"][idx],
        logs["train_accuracy"][idx],
        logs["val_accuracy"][idx],
    )
    print("LOGS:")
    print(log.format(*params))

    load_model(model, best_epoch, conf.path_model, CONF.device)

    train_loader = DataLoader(
        train_dataset, batch_size=CONF.loader.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        test_dataset, batch_size=CONF.loader.batch_size, shuffle=False
    )

    train_accuracy, train_loss = conf.eval_epoch(
        model, train_loader, conf.loss_fn, CONF.device
    )
    val_accuracy, val_loss = conf.eval_epoch(
        model, val_loader, conf.loss_fn, CONF.device
    )

    params = (
        strftime("%Y-%m-%d %H:%M:%S", gmtime(time())),
        best_epoch,
        train_loss,
        val_loss,
        train_accuracy * 100,
        val_accuracy * 100,
    )
    print("MODEL:")
    print(log.format(*params))


def pos_neg_dataset(CONF):
    transform = transforms.Compose(
        [
            transforms.RandomCrop(300),
            # transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    pos_dataset = My_Dataset(
        "pos",
        CONF.dataset.data_pos,
        augmentation=CONF.dataset.pos_aug,
        transform=transform,
        threshold=CONF.dataset.threshold,
        seed=CONF.seed,
        deterministic=True,
    )
    print(f"Positive dataset: {len(pos_dataset)}    ({pos_dataset.real_len})")

    neg_dataset = My_Dataset(
        "neg",
        CONF.dataset.data_neg,
        augmentation=CONF.dataset.neg_aug,
        transform=transform,
        threshold=CONF.dataset.threshold,
        seed=CONF.seed,
        deterministic=True,
    )
    print(f"Negative dataset: {len(neg_dataset)}    ({neg_dataset.real_len})")

    return pos_dataset, neg_dataset


def main(CONF):
    # if __name__ == "__main__":
    pos_dataset, neg_dataset = pos_neg_dataset(CONF)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # print(f"num parameters ResNet = {num_param}")
    # num_layers = len(list(model.children()))
    # print(f"num layers = {num_layers}")
    num_classes = 1000
    num_non_freeze = 513000
    print(f"num_non_freeze = {num_non_freeze}/{number_of_parameters(model)}")

    #############################
    seed_everything(CONF.seed)
    model = create_model(model, num_non_freeze, num_classes).to(CONF.device)
    print(f"num parameters ResNet = {number_of_parameters(model)}")

    dataset = TripletDataset(
        pos_dataset, neg_dataset, required_len=1000, deterministic=True, seed=CONF.seed
    )

    print("ResNet loop started!!!")
    common_train(CONF, model, dataset, kind="siam")

    print("ResNet loop done!!!")

    dataset = torch.utils.data.ConcatDataset([pos_dataset, neg_dataset])
    emb_dataset = Emb_Dataset(model, dataset, CONF.device)

    seed_everything(CONF.seed)
    cl = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(), nn.Linear(512, 2)).to(
        CONF.device
    )

    num_param = number_of_parameters(cl)
    print(f"num parameters = {num_param}")

    print("Classifier loop started!!!")
    common_train(CONF, cl, emb_dataset, kind="cl")
    print("Classifier loop done!!!")

    return pos_dataset, neg_dataset, model, cl
