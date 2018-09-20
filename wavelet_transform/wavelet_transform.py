import numpy as np
# from scipy import signal
from matplotlib.pyplot import imread


from wavelets import gabor2d, gaussian2d
from convolution import fast_convolve
from wavelets import show_wave # Unused but convenient for debugging


class Wavelet_Transform:
    """
    ********* Description *********
    Computes a wavelet transform on images
    ********* Attributes *********
    size : (int) = 31 : wavelets and gaussians array size
    scales : [(float)] = [2, 4, 8] : wavelets and gaussians scales
    angles : [(float)] = [0, 0.5, 1, 1.6, 2.1, 2.6] : wavelets and gaussians directions in pi-radians
    grid : ((int)) = () : grid for average (ex : (2,2,3) will return an array of size (2,2,3,*))
    kernels : (np array) : wavelets and gaussians computed
    ********* Return *********
    (np array(grid.shape,*)) : the wavelet tranform computed
    ********* Examples *********
    w = Wavelet_Transform()
    w = Wavelet_Transform(size=31, scales=[2,4,8], angles=np.arange(3)/.3*np.pi)
    result = w.forward("input/horses.jpg")
    result = w.forward("input/horses.jpg", mode="same")
    w.update_kernels(grid=(4,4,1))
    from scipy import misc
    img = misc.face()
    result = w.forward([img, "input/horses.jpg"])
    result = w.forward([img, "input/horses.jpg"], mode="valid", mean_or_max="max")
    """

    
    def __init__(self, size=63, scales=2**(1+np.arange(3)), angles=np.arange(6)/.6*np.pi, grid=()):
        self.size = None
        self.scales = None
        self.angles = None
        self.grid = None
        self.kernels = None
        self.update_kernels(size, scales, angles, grid)


        
    def update_kernels(self, size=None, scales=None, angles=None, grid=None):
        if size is not None:
            self.size = size
        if scales is not None:
            self.scales = scales
        if angles is not None:
            self.angles = angles
        if grid is not None:
            self.grid = grid
        self.kernels = np.concatenate([gaussian2d(self.size, self.scales), gabor2d(self.size, self.scales, 2, self.angles, zeromean=True)], axis=2)
        

        
    def forward(self, images="input/", output="stdout", mode='valid', mean_or_max="mean"):
        """
        ### TODO Allow images and output to be a directory ###
        ********* Description *********
        Performs the wavelet transform
        ********* Params *********
        images : (str) of (np array) or [(str) of (np array)] = "input/" : image(s) to compute
        output : (str) = "stdout" : where to put the output, either files or standard output
        mode : "valid" or "same" or "full" = "valid" : convolution padding
        mean_or_max : "mean" or "max" = "mean" : how to merge cells when the grid is defined
        """
        if (type(images) != list) and (type(images) != tuple):
            if (type(images) == str):
                img = imread(images)
            else:
                img = images
            c = fast_convolve(img, self.kernels, mode=mode)
            if self.grid != ():
                gr = (self.grid + img.shape[len(self.grid):])[:len(img.shape)]
                # average when (g in gr) = 1
                if (mean_or_max == "mean"):
                    c = np.mean(np.abs(c), axis=(tuple([ig for ig,g in enumerate(gr) if g == 1])))
                else:
                    c = np.max(np.abs(c), axis=(tuple([ig for ig,g in enumerate(gr) if g == 1])))
                gr = tuple([g for g in gr if g != 1])
                if (mean_or_max == "mean"):
                    for ig,g in enumerate(gr):
                        if (g < c.shape[ig]):
                            c = np.concatenate([np.mean(b, axis=ig, keepdims=True) for b in np.array_split(c, g, axis=ig)], axis=ig)
                else:
                    for ig,g in enumerate(gr):
                        if (g < c.shape[ig]):
                            c = np.concatenate([np.max(b, axis=ig, keepdims=True) for b in np.array_split(c, g, axis=ig)], axis=ig)
            if (output == "stdout"):
                return c
        else:
            c = np.stack([self.forward(img, output, mode) for img in images], axis=-1)
            if (output == "stdout"):
                return c
