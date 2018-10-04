import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np



def plot_error(error, x = "i", linecolor='.-k'):
    if (x == "i"):
        ind = 0
        plt.xlabel("nbr images")
    else:
        ind = 1
        plt.xlabel("time (s)")
    plt.ylabel("error")
    plt.plot([e[ind] for e in error], [e[2] for e in error], linecolor)



def show_weights(layer, transform=None):
    arr = layer.weight.detach().numpy()
    if transform is None:
        arr = arr + 0.5
    else:
        arr = transform(arr)
    ni = arr.shape[0]
    f, axarr = plt.subplots(ni)
    for i in range(ni):
        axarr[i].imshow(np.moveaxis(arr[i,:,:,:], 0, -1))
    plt.show()



def stack_filters(arr, axis = "x", separation = 1, move_axis=True):
    sh = list(arr.shape)
    x = 3
    if (axis == "y"):
        x = 2
    sh[x] = separation
    sep = np.ones(sh) / 2
    arr2 = np.concatenate([arr, sep], axis = x)
    arr3 = np.concatenate(arr2, axis=x-1)
    if (x == 3):
        arr4 = arr3[:,:,:-1]
    else:
        arr4 = arr3[:,:-1,:]
    if move_axis:
        arr5 = np.moveaxis(arr4, 0, -1)
        return arr5
    else:
        return arr4



def show_data_images(loader, n, classes=None):
    if isinstance(n, int):
        n = [n]
    n.sort()
    i = 0
    iter_images = iter(loader)
    while (i <= n[-1]):
        img, label = next(iter_images)
        for j in range(label.shape[0]):
            if i in n:
                plt.imshow(np.moveaxis(img[j].numpy(), 0, -1))
                if classes is None:
                    plt.title(int(label[j]))
                else:
                    plt.title(classes[int(label[j])])
                plt.show()
            i += 1
