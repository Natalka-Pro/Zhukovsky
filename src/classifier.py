import torch
from torch.utils.data import DataLoader


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    for batch in dataloader:
        X_batch, y_batch = batch

        logits = model(X_batch.to(device))

        loss = loss_fn(logits, y_batch.to(device))

        loss.backward()  # backpropagation (вычисление градиентов)
        optimizer.step()  # обновление весов сети
        optimizer.zero_grad()  # обнуляем веса


def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    # эта строка запрещает вычисление градиентов
    with torch.no_grad():
        for batch in dataloader:

            # получаем текущий батч
            X_batch, y_batch = batch
            total += len(y_batch)

            # получаем ответы сети на картинки батча
            logits = model(X_batch.to(device))

            # вычисляем лосс на текущем батче
            loss = loss_fn(logits, y_batch.to(device))
            running_loss += loss.item()

            # вычисляем ответы сети как номера классов для каждой картинки
            y_pred = torch.argmax(logits, dim=1)

            # вычисляем количество правильных ответов сети в текущем батче
            correct += torch.sum(y_pred.cpu() == y_batch)

    # вычисляем итоговую долю правильных ответов
    accuracy = correct / total
    return accuracy.numpy(), running_loss / len(dataloader)


def get_predictions(model, dataset, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    pred_items = []
    prob_items = []
    true_items = []
    x_items = []

    # эта строка запрещает вычисление градиентов
    with torch.no_grad():
        for batch in dataloader:

            # получаем текущий батч
            X_batch, y_batch = batch

            x_items.append(X_batch)
            true_items.append(y_batch)

            # получаем ответы сети на картинки батча
            logits = model(X_batch.to(device))

            # вычисляем ответы сети как номера классов для каждой картинки
            y_pred = torch.argmax(logits, dim=1)
            y_prob = torch.softmax(logits, dim=1)[:, 1]

            pred_items.append(y_pred)
            prob_items.append(y_prob)

    pred_items = torch.cat(pred_items)
    prob_items = torch.cat(prob_items)
    true_items = torch.cat(true_items)
    x_items = torch.cat(x_items)
    return x_items, true_items, pred_items.cpu(), prob_items.cpu()
