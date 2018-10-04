import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from imageio import imread, imwrite
from shutil import copy2



def transform_dataset(in_path, out_path, transform=None):
    if not os.path.isdir(in_path):
        print("Error : could not found the dataset at {}".format(in_path))
        raise IOError
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    if not os.path.isdir(out_path):
        print("Error : could not create root dataset path at {}".format(out_path))
        raise IOError
    list_files = os.listdir(in_path)
    for f in list_files:
        in_file = os.path.join(in_path, f)
        out_file = os.path.join(out_path, f)
        if os.path.isdir(in_file):
            transform_dataset(in_file, out_file, transform)
        elif (f[-4:] == '.png') or (f[-4:] == '.jpg') or (f[-4:] == '.bmp') or (f[-5:] == '.jpeg'):
            if transform is None:
                copy2(in_file, out_file)
            else:
                in_img = imread(in_file)
                out_img = transform(in_img)
                imwrite(out_file, out_img)
        else:
            copy2(in_file, out_file)



def resize_dataset(in_path, out_path, size):
    if not isinstance(size, tuple) or (len(size) != 2):
        print("Error : size must be a tuple of lenth 2")
        raise ValueError
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(size=size),
                                    ToNumpy()])
    transform_dataset(in_path, out_path, transform=transform)



class ToNumpy(object):
    """Convert a ``PIL Image`` or ``tensor`` to numpy_image.
    Converts a PIL Image or tensor to numpy.ndarray (H x W x C) without changing the range.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or tensor): Image to be converted to numpy.ndarray.
        Returns:
            numpy.ndarray: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return np.moveaxis(pic.detach().numpy(), 0, -1)
        else:
            return np.asarray(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'
