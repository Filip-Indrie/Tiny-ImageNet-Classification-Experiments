from train_utils import *
from models import *
import torch

if __name__ == "__main__":

    alexnet = AlexNet()
    vgg11 = VGG11()
    resnet18 = ResNet18()
    resnet34 = ResNet34()
    scope2 = Scope2()
    scope3 = Scope3()

    # print(alexnet)
    # print(vgg11)
    # print(resnet18)
    print(resnet34)
    print(scope2)
    print(scope3)



    batch_size = 64
    num_epochs = 1
    patience = 10
    weight_init = 'xavier_uniform'
    device = try_gpu()
    print(f"Training on {device}")

    lr_scope2 = 0.1
    lr_scope3 = 0.1
    lr_resnet34 = 0.1

    optimizer_scope2 = torch.optim.SGD(scope2.parameters(), lr=lr_scope2, weight_decay=0)
    optimizer_scope3 = torch.optim.SGD(scope3.parameters(), lr=lr_scope3, weight_decay=0)
    optimizer_resnet34 = torch.optim.SGD(resnet34.parameters(), lr=lr_resnet34, weight_decay=0)

    """
    The inputs of the loss function differs from function to function.
    Cross entropy loss takes a (batch_size, num_classes) tensor (predictions)
    and a (batch_size,) tensor (targets).
    The targets from DataLoader already come in a (batch_size,) tensor.
    """

    loss = torch.nn.CrossEntropyLoss()

    train_iter, val_iter = load_data_tiny_imagenet(batch_size)

    print("\nResNet34")
    train(resnet34, train_iter, val_iter, num_epochs, patience, loss, optimizer_resnet34, weight_init, device, delete_old_measurements=True)

    print("\nScope2")
    train(scope2, train_iter, val_iter, num_epochs, patience, loss, optimizer_scope2, weight_init, device, delete_old_measurements=True)

    print("\nScope3")
    train(scope3, train_iter, val_iter, num_epochs, patience, loss, optimizer_scope3, weight_init, device, delete_old_measurements=True)
