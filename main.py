from train_utils import *
from models import *
import torch

if __name__ == "__main__":

    alexnet = AlexNet()
    vgg11 = VGG11()
    resnet18 = ResNet18()

    batch_size = 128
    num_epochs = 1
    patience = 10
    weight_init = 'xavier_uniform'
    device = try_gpu()
    print(f"Training on {device}")

    lr_alexnet = 0.1
    lr_vgg11 = 0.1
    lr_resnet18 = 0.1

    optimizer_alexnet = torch.optim.SGD(alexnet.parameters(), lr=lr_alexnet)
    optimizer_vgg11 = torch.optim.SGD(vgg11.parameters(), lr=lr_vgg11)
    optimizer_resnet18 = torch.optim.SGD(resnet18.parameters(), lr=lr_resnet18)

    """
    The inputs of the loss function differs from function to function.
    Cross entropy loss takes a (batch_size, num_classes) tensor (predictions)
    and a (batch_size,) tensor (targets).
    The targets from DataLoader already come in a (batch_size,) tensor.
    """

    loss = torch.nn.CrossEntropyLoss()

    train_iter, val_iter = load_data_tiny_imagenet(batch_size)

    print("\nAlexNet")
    train(alexnet, train_iter, val_iter, num_epochs, patience, loss, optimizer_alexnet, weight_init, device, delete_old_measurements=True)
    train(alexnet, train_iter, val_iter, num_epochs, patience, loss, optimizer_alexnet, weight_init, device)
    train(alexnet, train_iter, val_iter, num_epochs, patience, loss, optimizer_alexnet, weight_init, device)

    print("VGG11")
    train(vgg11, train_iter, val_iter, num_epochs, patience, loss, optimizer_vgg11, weight_init, device, delete_old_measurements=True)

    print("\nResNet18")
    train(resnet18, train_iter, val_iter, num_epochs, patience, loss, optimizer_resnet18, weight_init, device, delete_old_measurements=True)
