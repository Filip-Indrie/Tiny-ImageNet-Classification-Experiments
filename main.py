from train_utils import *
from models import *
import torch

torch.manual_seed(42)

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
    num_epochs = 100
    patience = 10
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
    print(f"Training on {torch.cuda.get_device_name(device)}")

    weight_inits = ["default", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]
    optimizers = ["sgd", "sgd_nestrov", "adam", "adamw"]

    combos_used = []

    for optimizer in optimizers:
        for weight_init in weight_inits:

            print(optimizer, weight_init)
            resnet50 = ResNet50()
            opt = get_optimizer(resnet50, optimizer)
            train(resnet50, train_iter, val_iter, num_epochs, patience, loss, opt, weight_init, device)
