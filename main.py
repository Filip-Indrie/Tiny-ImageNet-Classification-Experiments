from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
import torchsummary
from train_utils import load_data_tiny_imagenet

net = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 32 * 32, 4096),
    nn.Linear(4096, 4096),
    nn.Linear(4096, 200)
)

batch_size = 128
lr = 1
num_epochs = 10
patience = 10

train_iter, val_iter = load_data_tiny_imagenet(batch_size)

x = torch.randn(1, 3, 64, 64)
print(net(x).shape)

# train_loss_all, train_acc_all, val_loss_all, val_acc_all = train(net, train_iter, val_iter, num_epochs, lr, patience, try_gpu())