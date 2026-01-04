from train_utils import *
from models import *
import torch

if __name__ == "__main__":

    nets = [
        AlexNet(),
        VGG11(),
        ResNet18(),
        ResNet34(),
        ResNet50(),
        Scope2(),
        Scope3(),
        Scope2Atrous(),
        Scope3Atrous(),
        ShallowBottleNet(),
        BottleNet(),
        DeepBottleNet(),
        DeeperBottleNet()
    ]



    batch_size = 64
    num_epochs = 1
    patience = 10
    weight_init = 'xavier_uniform'
    lr = 0.1

    """
    The inputs of the loss function differs from function to function.
    Cross entropy loss takes a (batch_size, num_classes) tensor (predictions)
    and a (batch_size,) tensor (targets).
    The targets from DataLoader already come in a (batch_size,) tensor.
    """

    loss = torch.nn.CrossEntropyLoss()

    train_iter, val_iter = load_data_tiny_imagenet(batch_size)

    device = try_gpu()
    print(f"Training on {device}")

    for net in nets:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0)
        print(type(net).__name__)
        train(net, train_iter, val_iter, num_epochs, patience, loss, optimizer, weight_init, device)
