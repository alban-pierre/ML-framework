import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import os
from imageio import imread



class One_Block_Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return min(self.data.shape[0], self.labels.shape[0])

    def __getitem__(self, idx):
        # return {'image': self.data[idx], 'landmarks': self.labels[idx]}
        return (self.data[idx], self.labels[idx])



class Images_Dataset(Dataset):
    def __init__(self, img_path, labels, transform=None):
        self.img_path = np.asarray(img_path)
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return min(self.img_path.shape[0], self.labels.shape[0])

    def __getitem__(self, idx):
        img = imread(self.img_path[idx])
        if (img.shape[2] == 4):
            img = img[:,:,:3]
        if self.transform:
            return (self.transform(img), self.labels[idx])
        else:
            return (g, self.labels[idx])



class Train_Classifier_Dataset(Images_Dataset):
    def __init__(self, datapath, transform=None):
        self.datapath = datapath
        self.transform = transform
        self.classes = os.listdir(datapath)
        self.img_path = []
        self.labels = []
        for i,c in enumerate(self.classes):
            list_img = os.listdir(os.path.join(datapath, c))
            self.img_path += [os.path.join(datapath, c, l) for l in list_img]
            self.labels += [i for l in list_img]
        self.img_path = np.asarray(self.img_path)
        self.labels = np.asarray(self.labels)



def load_cifar(no_transform=False):
    if no_transform:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../datasets/CIFAR10/',
                                            train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../datasets/CIFAR10/',
                                           train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, testset, trainloader, testloader, classes



def load_plant_seed(size=(32,32), no_transform=False):
    if no_transform:
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = Train_Classifier_Dataset("../datasets/plant_seed/train/", transform=transform)
    elif (size == (224,224)):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = Train_Classifier_Dataset("../datasets/plant_seed_224/train/", transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.ToPILImage(),
                                        transforms.Resize(size=size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = Train_Classifier_Dataset("../datasets/plant_seed/train/", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = None
    testloader = None
    classes = trainset.classes
    return trainset, testset, trainloader, testloader, classes
