import torch
from train_utils import load_data_tiny_imagenet
from models import *

alexnet = AlexNet()
vgg11 = VGG11()
resnet18 = ResNet18()

batch_size = 128
lr = 1
num_epochs = 10
patience = 10

train_iter, val_iter = load_data_tiny_imagenet(batch_size)

x = torch.randn(1, 3, 128, 128)
print(alexnet(x).shape)
print(vgg11(x).shape)
print(resnet18(x).shape)

# train_loss_all, train_acc_all, val_loss_all, val_acc_all = train(net, train_iter, val_iter, num_epochs, lr, patience, try_gpu())