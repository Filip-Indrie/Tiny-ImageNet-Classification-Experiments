from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn

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

    transform_val = [transforms.ToTensor(),
                     normalize]

    transform_val = transforms.Compose(transform_val)

    imagenet_train = datasets.ImageFolder("../data/tiny-imagenet-200/train", transform=transform_train)
    imagenet_val = datasets.ImageFolder("../data/tiny-imagenet-200/val", transform=transform_val)

    train_loader = DataLoader(imagenet_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def evaluate_accuracy(net, data_iter, loss, device):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_hits = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)

            with torch.no_grad():
                total_loss += float(l)
                total_hits += sum([net(X).argmax(axis=1).type(y.dtype) == y])
                total_samples += y.numel()
    return float(total_loss) / len(data_iter), float(total_hits) / total_samples  * 100

def train_epoch(net, train_iter, loss, optimizer, device):
    # Set the model to training mode
    net.train()

    # Sum of training loss, sum of training correct predictions, no. of examples
    total_loss = 0
    total_hits = 0
    total_samples = 0

    for X, y in train_iter:
        # Compute gradients and update parameters
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)

        # Using PyTorch built-in optimizer & loss criterion
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += float(l)
            total_hits += sum([y_hat.argmax(axis=1).type(y.dtype) == y])
            total_samples += y.numel()
    # Return training loss and training accuracy
    return float(total_loss) / len(train_iter), float(total_hits) / total_samples  * 100

def train(net, train_iter, val_iter, num_epochs, patience, loss, optimizer, device):
    """Train a model."""
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []

    best_val_accuracy = 0
    counter = 0

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    print('Training on', device)
    net.to(device)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer, device)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)

        val_loss, val_acc = evaluate_accuracy(net, val_iter, loss, device)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)

        print(f'Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}')

        if(val_acc > best_val_accuracy):
            best_val_accuracy = val_acc
            counter = 0
        else:
            counter += 1

        if(counter > patience):
            break

    print(f'Best Validation Accuracy {best_val_accuracy:.2f}, Epoch: {val_acc_all.index(best_val_accuracy):.d}')

    return train_loss_all, train_acc_all, val_loss_all, val_acc_all

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')