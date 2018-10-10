import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import os
from imageio import imread

from utils.split_dataset import *
from utils.split_pytorch_dataset import *

import sys
this_file_path = '/'.join(__file__.split('/')[:-2])
full_path = os.path.join(os.getcwd(), this_file_path)
dataset_root_path = os.path.join(full_path, 'datasets/')



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
    def __init__(self, datapath, transform=None, size=None, indexes=None):
        self.datapath = datapath
        self.transform = transform
        self.classes = os.listdir(datapath)
        self.img_path = []
        self.labels = []
        self.indexes = indexes
        for i,c in enumerate(self.classes):
            list_img = os.listdir(os.path.join(datapath, c))
            self.img_path += [os.path.join(datapath, c, l) for l in list_img]
            self.labels += [i for l in list_img]
        self.img_path = np.asarray(self.img_path)
        self.labels = np.asarray(self.labels)
        if (size is not None) and (self.indexes is None):
            self.indexes, self.indexes_test = Uniform_Split_Dataset().split_indexes(self.labels, size)
        if (self.indexes is not None):
            self.img_path = self.img_path[self.indexes]
            self.labels = self.labels[self.indexes]





# from collections import Counter

# class Split_Dataset(Dataset):
#     def __init__(self, dataset, split=None, size=None, uniform=False):
#         self.dataset = dataset
#         self.split = split
#         self.size = size
#         self.class_uniform = None
#         if (self.split is None):
#             if (self.size is None):
#                 self.size = 0.5
#             if isinstance(self.size, float):
#                 self.size = int(round(len(self.dataset) * self.size))
#             self.get_split_indexes(uniform=uniform)


#     def get_split_indexes(self, uniform=True):
#         if (not uniform) and ((self.class_uniform is None) or self.class_uniform):
#             self._compute_split_indexes()
#         if (uniform) and (not self.class_uniform):
#             self._compute_split_indexes_uniform()
#         return self.split


#     def _compute_split_indexes(self):
#         self.corresp = np.arange(len(self.dataset))
#         np.random.shuffle(self.corresp)
#         self.corresp = self.corresp[:self.size]
#         self.corresp.sort()
#         self._compute_split_from_corresp()


#     def _compute_split_indexes_uniform(self):
#         labels = [i[1] for i in self.dataset]
#         count = dict(Counter(labels))
#         s = self.size
#         ts = len(labels)
#         ls = {}
#         for k,v in count.items():
#             fv = v*(float(s)/ts)
#             ls[k] = int(fv) + int((np.random.rand(1)[0] < fv - int(fv)))
#             ts -= v
#             s -= ls[k]
#         self.corresp = []
#         for k,v in ls.items():
#             indexes = np.asarray([i for i,j in enumerate(labels) if j == k])
#             np.random.shuffle(indexes)
#             self.corresp.append(indexes[:v])
#         self.corresp = np.concatenate(self.corresp)
#         self.corresp.sort()
#         self._compute_split_from_corresp()


#     def _compute_split_from_corresp(self):
#         self.split = np.zeros(len(self.dataset)).astype(int)
#         for i in self.corresp:
#             self.split[i] = 1


#     def __len__(self):
#         return self.corresp.shape[0]


#     def __getitem__(self, idx):
#         return self.dataset[self.corresp[idx]]
            
            



def load_cifar(no_transform=False):
    if no_transform:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_root_path, 'CIFAR10/'),
                                            train=True, download=True, transform=transform)
    valset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_root_path, 'CIFAR10/'),
                                           train=False, download=True, transform=transform)
    testset = None
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False, num_workers=2)
    testloader = None
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, valset, testset, trainloader, valloader, testloader, classes



def load_plant_seed(size=(32,32), no_transform=False, validation_size=None):
    dataset_path = 'plant_seed/train/'
    if no_transform:
        transform = transforms.Compose([transforms.ToTensor()])
    elif (size == (224,224)):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_path = 'plant_seed_224/train/'
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.ToPILImage(),
                                        transforms.Resize(size=size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Now that the path and transforms are defined, we can load the dataset
    fpath = os.path.join(dataset_root_path, dataset_path)
    trainset = Train_Classifier_Dataset(fpath, transform=transform, size=validation_size)
    # If we create a validation set inside the train set
    valset = None
    if (validation_size is not None):
        valset = Train_Classifier_Dataset(fpath, transform=transform, indexes=trainset.indexes_test)
    testset = None
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    valloader = None
    if valset is not None:
        valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True, num_workers=2)
    testloader = None
    classes = trainset.classes
    return trainset, valset, testset, trainloader, valloader, testloader, classes
