from train_utils import *
from models import *


alexnet = AlexNet(num_classes=200)
vgg11 = VGG11(num_classes=200)
resnet18 = ResNet18(num_classes=200)

batch_size = 128
lr = 1 # is set in the optimizer, not the training function
num_epochs = 1
patience = 10

optimizer_alexnet = torch.optim.SGD(alexnet.parameters(), lr=lr)
optimizer_vgg11 = torch.optim.SGD(vgg11.parameters(), lr=lr)
optimizer_resnet18 = torch.optim.SGD(resnet18.parameters(), lr=lr)

"""
    The inputs of the loss function differs from function to function.
    Cross entropy loss takes a (batch_size, num_classes) tensor (predictions)
    and a (batch_size,) tensor (targets).
    The targets from DataLoader already come in a (batch_size,) tensor.
"""
loss = torch.nn.CrossEntropyLoss()

train_iter, val_iter = load_data_tiny_imagenet(batch_size)

train_loss_alexnet, train_acc_alexnet, val_loss_alexnet, val_acc_alexnet = train(alexnet, train_iter, val_iter, num_epochs, patience, loss, optimizer_alexnet, try_gpu())
train_loss_vgg11, train_acc_vgg11, val_loss_vgg11, val_acc_vgg11 = train(alexnet, train_iter, val_iter, num_epochs, patience, loss, optimizer_alexnet, try_gpu())
train_loss_resnet18, train_acc_resnet18, val_loss_resnet18, val_acc_resnet18 = train(alexnet, train_iter, val_iter, num_epochs, patience, loss, optimizer_alexnet, try_gpu())