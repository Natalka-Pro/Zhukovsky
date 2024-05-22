import torch
# https://habr.com/ru/articles/794750/


def train_siam(model, train_loader, val_loader, loss_fn, optimizer, n_epoch, globals=None):
    DEVICE = globals["DEVICE"]

    log = f"# Epoch {{:{len(str(n_epoch))}}}/{n_epoch} "
    log += f"train/val: loss {{:6.5f}}/{{:6.5f}}, accuracy: {{:6.4f}}%/{{:6.4f}}%"

    for epoch in range(n_epoch):
        train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)

        train_loss, train_accuracy = validate_epoch(model, train_loader, loss_fn, DEVICE)
        val_loss, val_accuracy     = validate_epoch(model, val_loader, loss_fn, DEVICE)

        print(
            log.format(epoch + 1,
                       train_loss, 
                       val_loss,
                       train_accuracy * 100,
                       val_accuracy * 100))
        
    return model


def train_epoch(model, dataloader, loss_fn, optimizer, device):

    model.train()
    # running_loss = 0.0

    for data in dataloader:
        anchor, positive, negative = [d.to(device) for d in data]

        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        loss = loss_fn(anchor_output, positive_output, negative_output)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # running_loss += loss.item()

    # return running_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            anchor, positive, negative = [d.to(device) for d in data]
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = loss_fn(anchor_output, positive_output, negative_output)
            running_loss += loss.item()

            total += anchor.size(0)
            correct += (torch.norm(anchor_output - positive_output, dim=1) <
                        torch.norm(anchor_output - negative_output, dim=1)).sum().item()

    accuracy = correct / total
    return running_loss / len(dataloader), accuracy


