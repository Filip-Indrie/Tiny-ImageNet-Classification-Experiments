from train_utils import *
from models import *
import torch

torch.manual_seed(50)

if __name__ == "__main__":

    """
        Nets is defined as a tuple of a network and its associated learning rate.
        If the learning rate is None, the default value (1e-3) is used.
    """
    nets = [
        # (AlexNet(), None),
        # (VGG11(), None),
        # (ResNet18(), None),
        # (ResNet34(), None),
        # (ResNet50(), None),
        # (Scope2(), None),
        # (Scope3(), None),
        # (Scope2Atrous(), None),
        # (Scope3Atrous(), None),
        # (ShallowBottleNet(), None),
        # (BottleNet(), None),
        # (DeepBottleNet(), None),
        # (DeeperBottleNet(), None),
        # (DilatedHeadNet(), None),
        # (MultiHeadNet(), None),
        # (ShallowMultiHeadNet(), None),
        # (StandardTransformer(), 1e-2),
        # (DeepTransformer(), 1e-2),
        # (WideTransformer(), 1e-3),
        # (DeepWideTransformer(), 1e-3),
        # (WideTransformerV2(), 1e-3),
        # (DeepWideTransformerV2(), 1e-3),
        # (WideTransformerV3(), 1e-3),
        # (DeepWideTransformerV3(), 1e-3),
        # (LowResTransformer(), 1e-3),
        # (DeepLowResTransformer(), 1e-3),
        # (WideLowResTransformer(), 1e-3),
        # (DeepWideLowResTransformer(), 1e-3),
        # (DeeperWideLowResTransformer(), 1e-3),
        # (WideLowResTransformerV2(), 1e-3),
        # (DeepWideLowResTransformerV2(), 1e-3),
        # (DeeperWideLowResTransformerV2(), 1e-3),
        # (CNNViT(), 1e-3),
        # (CNNViTNoBottleneck(), 1e-3),
        # (LowResCNNViT(), 1e-3),
        # (LowResCNNViTNoBottleneck(), 1e-3),
        # (LowResCNNViTV2(), 1e-4),
        # (LowResCNNViTV3(), 1e-4),
        (CNNViTNoPatches(), 1e-3),
        (CNNViTNoPatchesV2(), 1e-3),
        (CNNViTNoPatchesV3(), 1e-3),
        (LowResCNNViTNoPatches(), 1e-3),
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

    for net, lr in nets:
        if lr is None:
            opt = get_optimizer(net, optimizer)
        else:
            opt = get_optimizer(net, optimizer, lr=lr)

        print(type(net).__name__)
        # print(f"Learning rate: {opt.state_dict()['param_groups'][0]['lr']}")
        train(net, train_iter, val_iter, num_epochs, patience, loss, opt, weight_init, device)

