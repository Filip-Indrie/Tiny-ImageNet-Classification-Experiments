from train_utils import *
from models import *
import torch

torch.manual_seed(50)

if __name__ == "__main__":

    nets = [
        # AlexNet(),
        # VGG11(),
        # ResNet18(),
        # ResNet34(),
        # ResNet50(),
        # Scope2(),
        # Scope3(),
        # Scope2Atrous(),
        # Scope3Atrous(),
        # ShallowBottleNet(),
        # BottleNet(),
        DeepBottleNet(),
        DeeperBottleNet()
    ]

    batch_size = 64
    num_epochs = 100
    patience = 10

    loss = torch.nn.CrossEntropyLoss()

    train_iter, val_iter = load_data_tiny_imagenet(batch_size)

    device = try_gpu()
    print(f"Training on {torch.cuda.get_device_name(device)}")

    """
        Through experiments, it is to be observed that the combination of
        AdamW optimizer and kaiming uniform weight initialization yields
        one of the best accuracy percentage the fastest. (52.64% at epoch 13 on ResNet50).
    """

    optimizer = "adamw"
    weight_init = "kaiming_uniform"

    for net in nets:
        opt = get_optimizer(net, optimizer)
        print(type(net).__name__)
        train(net, train_iter, val_iter, num_epochs, patience, loss, opt, weight_init, device)