from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
import time
import os

__all__ = ["load_data_tiny_imagenet", "init_weights", "evaluate_accuracy", "train_epoch", "train", "try_gpu"]

def load_data_tiny_imagenet(batch_size=128):
    """Load the tiny imagenet dataset."""

    # mean and std of ImageNet images
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform_train = [transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       normalize]

    transform_train = transforms.Compose(transform_train)

    transform_val = [transforms.Resize(128),
                     transforms.ToTensor(),
                     normalize]

    transform_val = transforms.Compose(transform_val)

    imagenet_train = datasets.ImageFolder("../data/tiny-imagenet-200/train", transform=transform_train)
    imagenet_val = datasets.ImageFolder("../data/tiny-imagenet-200/val", transform=transform_val)

    train_loader = DataLoader(imagenet_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader

def init_weights_random(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.uniform_(m.weight)

def init_weights_xavier_uniform(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def init_weights_xavier_normal(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)

def init_weights_kaiming_uniform(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight)

def init_weights_kaiming_normal(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)

def init_weights(mode):
    match mode:
        case 'random': return init_weights_random
        case 'xavier_uniform': return init_weights_xavier_uniform
        case 'xavier_normal': return init_weights_xavier_normal
        case 'kaiming_uniform': return init_weights_kaiming_uniform
        case 'kaiming_normal': return init_weights_kaiming_normal
        case _: return init_weights_random

def evaluate_accuracy(net, data_iter, loss, device):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_hits = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)

            with torch.no_grad():
                total_loss += float(l)
                total_hits += (y_hat.argmax(axis=1).type(y.dtype) == y).sum()
                total_samples += y.numel()

    return float(total_loss) / len(data_iter), float(total_hits) / total_samples  * 100

def train_epoch(net, train_iter, loss, optimizer, device):
    # Set the model to training mode
    net.train()

    # Sum of training loss, sum of training correct predictions, no. of examples
    total_loss = 0
    total_hits = 0
    total_samples = 0

    for x, y in train_iter:
        # Compute gradients and update parameters
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        l = loss(y_hat, y)

        # Using PyTorch built-in optimizer & loss criterion
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += float(l)
            total_hits += (y_hat.argmax(axis=1).type(y.dtype) == y).sum()
            total_samples += y.numel()

    # Return training loss and training accuracy
    return float(total_loss) / len(train_iter), float(total_hits) / total_samples  * 100

def train(net, train_iter, val_iter, num_epochs, patience, loss, optimizer, weight_init,  device, delete_old_measurements=False):
    """Train a model."""
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    best_val_accuracy = 0
    counter = 0

    net.apply(init_weights(weight_init))

    net.to(device)

    net_name = type(net).__name__
    dir_name = "Measurements/" + net_name + "/"
    os.makedirs(dir_name, exist_ok=True)

    optimizer_name = type(optimizer).__name__
    loss_name = type(loss).__name__
    lr = str(optimizer.param_groups[0]['lr'])

    stats_file_name = dir_name + weight_init + "_" + loss_name + "_" + optimizer_name + "_" + lr + ".txt"
    if not os.path.exists(stats_file_name) or delete_old_measurements:
        stats_file = open(stats_file_name, "w", encoding="utf-8")
        stats_file.write(str(net) + "\n\n")
    else:
        stats_file = open(stats_file_name, "a", encoding="utf-8")
        stats_file.write("\n")

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")

        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer, device)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)

        val_loss, val_acc = evaluate_accuracy(net, val_iter, loss, device)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)

        stats_file.write(f'Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}\n')

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            counter = 0
        else:
            counter += 1

        if counter > patience:
            break

    end_time = time.time()

    stats_file.write(f'Best Validation Accuracy {best_val_accuracy:.2f}, Epoch: {val_acc_all.index(best_val_accuracy) + 1}, Training Time: {end_time - start_time:.2f}s\n')

    stats_file.close()

    return train_loss_all, train_acc_all, val_loss_all, val_acc_all

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')