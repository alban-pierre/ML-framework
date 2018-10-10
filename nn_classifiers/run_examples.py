import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from .load_data import *
from .nets.nn2c3f_classifier import Net
from .train_test_utils import *
from .visualization_utils import *


example_1 = False
example_2 = False
example_3 = False



if example_1:
    # trainset, testset, trainloader, testloader, classes = load_cifar()
    trainset, valset, testset, trainloader, valloader, testloader, classes = load_cifar()
    net = Net(n_out=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # train_monitors = [Train_Error(2500), Test_Error(testloader, 6250)]
    train_monitors = [Train_Error(2500), Test_Error(valloader, 6250)]
    res = train(net, trainloader, criterion, optimizer, nbr_epochs=1, train_monitors=train_monitors)
    train_monitors[0].plot(res)
    train_monitors[1].plot(res)
    plt.title("Train and test error (black & red)")
    plt.show()
    train_monitors[0].plot(res, x="t")
    train_monitors[1].plot(res, x="t")
    plt.title("Train and test error (black & red)")
    plt.show()



if example_2:
    trainset, valset, testset, trainloader, valloader, testloader, classes = load_cifar()
    net = Net(n_out=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    res = {}
    # testpred = Test_Prediction(testloader)
    testpred = Test_Prediction(valloader)
    train_monitors = [Train_Error(1250),
                      Running_Loss(1250),
                      Test_Error(testpred, 2500),
                      Test_Error_Per_Class(testpred, 5000),
                      Test_False_Positive_Per_Class(testpred, 5000),
                      Net_Parameters_Change(5000),
                      Show_Conv_Filters(2500)]
    res = train(net, trainloader, criterion, optimizer, nbr_images=10000, train_monitors=train_monitors, resume=res)
    train_monitors[0].plot(res)
    train_monitors[2].plot(res)
    train_monitors[3].plot(res)
    train_monitors[4].plot(res)
    plt.title("Train and test error (black & red)")
    plt.ylabel("error")
    plt.xlabel("nbr images")
    plt.show()
    train_monitors[1].plot(res)
    plt.show()
    train_monitors[5].plot(res)
    plt.show()
    train_monitors[6].plot(res)
    plt.show()
    # You can still have access to test_errors etc by doing "te, _ = Test_Error(testpred)(net)"
    # Note that if you ask again, it will not recompute the net output
    # To force recomputation, do "te, _ = Test_Error(testpred)(net, recompute=True)"




if example_3:
    from torchvision.models.vgg import vgg11
    trainset, valset, testset, trainloader, valloader, testloader, classes = load_plant_seed(size=(224, 224), validation_size=10)
    net = vgg11(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False
    #net.fc3 = nn.Linear(4096, len(classes))
    net.classifier[6] = nn.Linear(4096, len(classes))
    freeze(net, -1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    res = {}
    testpred = Test_Prediction(valloader)
    train_monitors = [Train_Error(10),
                      Running_Loss(10),
                      Test_Error(testpred, 50)]#,
                      # Test_Error_Per_Class(testpred, 5000),
                      # Test_False_Positive_Per_Class(testpred, 5000),
                      # Net_Parameters_Change(5000),
                      # Show_Conv_Filters(2500)]
    res = train(net, trainloader, criterion, optimizer, nbr_images=200, train_monitors=train_monitors, resume=res)
 
