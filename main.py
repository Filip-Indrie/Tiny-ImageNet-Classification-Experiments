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
    weight_init = 'xavier_normal'
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

    # print(type(str(alexnet)))
    # print(vgg11)
    # print(resnet18)

    print("\nAlexNet")
    train_loss_alexnet, train_acc_alexnet, val_loss_alexnet, val_acc_alexnet = train(alexnet, train_iter, val_iter, num_epochs, patience, loss, optimizer_alexnet, weight_init, device)

    # print("VGG11")
    # train_loss_vgg11, train_acc_vgg11, val_loss_vgg11, val_acc_vgg11 = train(vgg11, train_iter, val_iter, num_epochs, patience, loss, optimizer_vgg11, weight_init, device)
    #
    print("\nResNet18")
    train_loss_resnet18, train_acc_resnet18, val_loss_resnet18, val_acc_resnet18 = train(resnet18, train_iter, val_iter, num_epochs, patience, loss, optimizer_resnet18, weight_init, device)
