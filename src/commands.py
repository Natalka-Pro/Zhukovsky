from hydra import compose, initialize

from torchvision import datasets, models, transforms
from time import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import My_Dataset, TripletDataset, Emb_Dataset
from .functions import number_of_parameters, seed_everything, create_model
from .dataset_fun import split_dataset
from .train_siam import train_siam
from .train import train as train_cl


def hydra_config():
    initialize(config_path=".", version_base="1.3")
    config = compose(config_name="config.yaml")
    return config


def loop(model, CONF, dataset, margin, train=True, kind="siam"):
    seed_everything(CONF.seed)
    train_dataset, test_dataset = split_dataset(dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=CONF.loader.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONF.loader.batch_size, shuffle=False
    )

    start_time = time()
    seed_everything(CONF.seed)

    if kind == "siam":
        loss = nn.TripletMarginLoss(margin=margin, p=2)
        train_fun = train_siam
        path = CONF.siam.path
        n_epochs = CONF.siam.n_epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=CONF.siam.learning_rate)

    elif kind == "cl":
        loss = torch.nn.CrossEntropyLoss()
        train_fun = train_cl
        path = CONF.classifier.path
        n_epochs = CONF.classifier.n_epochs
        optimizer = torch.optim.Adam(
            model.parameters(), lr=CONF.classifier.learning_rate
        )

    if train:
        model = train_fun(
            model, train_loader, test_loader, loss, optimizer, n_epochs, CONF.device
        )
        torch.save(model.state_dict(), path)
    else:
        model.load_state_dict(torch.load(path, map_location=CONF.device))

    print(f"# Время работы: {(time() - start_time):6.5f}s")
    return model


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


def main(CONF, margin, train=True):
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
    model = loop(model, CONF, dataset, margin, train=train, kind="siam")
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
    cl = loop(cl, CONF, emb_dataset, margin, train=train, kind="cl")
    print("Classifier loop done!!!")

    return pos_dataset, neg_dataset, model, cl
