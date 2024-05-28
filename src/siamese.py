import torch


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    # https://habr.com/ru/articles/794750/
    model.train()

    for batch in dataloader:
        anchor, positive, negative = [d.to(device) for d in batch]

        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        loss = loss_fn(anchor_output, positive_output, negative_output)

        loss.backward()  # backpropagation (вычисление градиентов)
        optimizer.step()  # обновление весов сети
        optimizer.zero_grad()  # обнуляем веса


def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            anchor, positive, negative = [d.to(device) for d in batch]
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = loss_fn(anchor_output, positive_output, negative_output)
            running_loss += loss.item()

            total += anchor.size(0)
            correct += (
                (
                    torch.norm(anchor_output - positive_output, dim=1)
                    < torch.norm(anchor_output - negative_output, dim=1)
                )
                .sum()
                .item()
            )

    accuracy = correct / total
    return accuracy, running_loss / len(dataloader)
