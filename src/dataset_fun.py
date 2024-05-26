import torch


def pos_neg(dataloader):
    num_pos = num_neg = num = 0

    for batch in dataloader:
        images, labels = batch

        col = images.size(0)
        pos = labels.sum()

        num += col
        num_pos += pos
        num_neg += col - pos

    return {"1": num_pos.item(), "0": num_neg.item(), "total": num}


def split_dataset(dataset, percent=0.8):
    # в тренировочную выборку отнесем 80% всех картинок
    train_size = int(len(dataset) * percent)
    # в валидационную — остальные 20%
    val_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
                        dataset, [train_size, val_size])

    len_tr, len_test = len(train_dataset), len(test_dataset)
    print(f"split_dataset:\nTrain: {len_tr}\nTest: {len_test}\nTotal: {len_tr + len_test}")
    
    return train_dataset, test_dataset
