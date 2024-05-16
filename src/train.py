import numpy as np
import torch
from torch.utils.data import DataLoader



def get_predictions(model, dataset, globals=None):
    BATCH_SIZE = globals["BATCH_SIZE"]
    DEVICE = globals["DEVICE"]

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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
            logits = model(X_batch.to(DEVICE))
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


def train(model, train_loader, val_loader, loss_fn, optimizer, n_epoch, globals=None):
    DEVICE = globals["DEVICE"]

    # цикл обучения сети
    for epoch in range(n_epoch):
        
        model.train(True)

        # running_losses = []
        # running_accuracies = []
        for i, batch in enumerate(train_loader):
            # получаем текущий батч
            X_batch, y_batch = batch

            # forward pass (получение ответов на батч картинок)
            logits = model(X_batch.to(DEVICE))

            # вычисление лосса от выданных сетью ответов и правильных ответов на батч
            loss = loss_fn(logits, y_batch.to(DEVICE))
            # running_losses.append(loss.item())

            loss.backward()  # backpropagation (вычисление градиентов)
            optimizer.step()  # обновление весов сети
            optimizer.zero_grad()  # обнуляем веса

            # вычислим accuracy на текущем train батче
            # model_answers = torch.argmax(logits, dim=1)
            # train_accuracy = torch.sum(y_batch == model_answers.cpu()) / len(y_batch)
            # running_accuracies.append(train_accuracy)

            # Логирование результатов
            # if (i + 1) % 50 == 0:
            #     print(
            #         "Средние train лосс и accuracy на последних 50 итерациях:",
            #         np.mean(running_losses),
            #         np.mean(running_accuracies),
            #         end="\n",
            #     )

        # после каждой эпохи получаем метрику качества на валидационной выборке
        model.train(False)

        train_accuracy, train_loss = evaluate(model, train_loader, loss_fn, globals=globals)
        val_accuracy, val_loss = evaluate(model, val_loader, loss_fn, globals=globals)
        print(
            "# Epoch {}/{} train/val: loss {:6.5f}/{:6.5f}, accuracy: {:6.5f}/{:6.5f}".format(
            epoch + 1,
            n_epoch,
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            ),
        )

    return model


def evaluate(model, dataloader, loss_fn, globals=None):
    DEVICE = globals["DEVICE"]

    losses = []

    num_correct = 0
    num_elements = 0

    for i, batch in enumerate(dataloader):

        # получаем текущий батч
        X_batch, y_batch = batch
        num_elements += len(y_batch)

        # эта строка запрещает вычисление градиентов
        with torch.no_grad():
            # получаем ответы сети на картинки батча
            logits = model(X_batch.to(DEVICE))

            # вычисляем лосс на текущем батче
            loss = loss_fn(logits, y_batch.to(DEVICE))
            losses.append(loss.item())

            # вычисляем ответы сети как номера классов для каждой картинки
            y_pred = torch.argmax(logits, dim=1)

            # вычисляем количество правильных ответов сети в текущем батче
            num_correct += torch.sum(y_pred.cpu() == y_batch)

    # вычисляем итоговую долю правильных ответов
    accuracy = num_correct / num_elements

    return accuracy.numpy(), np.mean(losses)
