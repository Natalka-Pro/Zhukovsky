import os
import pickle
from time import gmtime, strftime, time

import torch
from tqdm import tqdm


def save_model(model, epoch, dir):
    path = os.path.join(dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), path)


def load_model(model, epoch, dir, device):
    path = os.path.join(dir, f"epoch_{epoch}.pth")
    model.load_state_dict(torch.load(path, map_location=device))


def save_logs(logs, path):
    with open(path, "wb") as f:
        pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_logs(path, verbose=False):
    with open(path, "rb") as f:
        logs = pickle.load(f)

    if verbose:
        for i in range(len(logs["train_loss"])):
            params = (
                logs["time"][i],
                logs["epoch"][i],
                logs["train_loss"][i],
                logs["val_loss"][i],
                logs["train_accuracy"][i],
                logs["val_accuracy"][i],
            )
            log = f"# {{}} Epoch {{:{logs['epoch'][-1]}}} "
            log += f"train/val: loss {{:6.5f}}/{{:6.5f}}, acc:{{:7.3f}}%/{{:7.3f}}%"
            print(log.format(*params))

    return logs


def train(
    model,
    train_loader,
    val_loader,
    train_epoch,
    eval_epoch,
    loss_fn,
    optimizer,
    n_epochs,
    device,
    kind,
    path_log,
    path_model,
):
    print(f"train: started, {kind = }")

    log = f"# {{}} Epoch {{:{len(str(n_epochs))}}} "
    log += f"train/val: loss {{:6.5f}}/{{:6.5f}}, acc:{{:7.3f}}%/{{:7.3f}}%"

    logs = {
        "time": [],
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    start_time = time()

    for epoch in range(n_epochs):
        train_epoch(model, train_loader, loss_fn, optimizer, device)

        train_accuracy, train_loss = eval_epoch(model, train_loader, loss_fn, device)
        val_accuracy, val_loss = eval_epoch(model, val_loader, loss_fn, device)

        params = (
            strftime("%Y-%m-%d %H:%M:%S", gmtime(time())),
            epoch + 1,
            train_loss,
            val_loss,
            train_accuracy * 100,
            val_accuracy * 100,
        )

        print(log.format(*params))

        logs["time"].append(params[0])
        logs["epoch"].append(params[1])
        logs["train_loss"].append(params[2])
        logs["val_loss"].append(params[3])
        logs["train_accuracy"].append(params[4])
        logs["val_accuracy"].append(params[5])

        save_model(model, epoch + 1, path_model)
        save_logs(logs, path_log)

    t = strftime("%H:%M:%S", gmtime(time() - start_time))
    print(f"# Время работы: {t}")
