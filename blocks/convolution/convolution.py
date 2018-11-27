import numpy as np
#from scipy import signal
#import matplotlib.pyplot as plt
from scipy.signal import fftconvolve



def fast_convolve(img, kernels, res=None, mode='same', vtype=np.float32, prefetching=False):
    """
    ********* Description *********
    Compute the convolution using FFTs
    ********* Params *********
    img : (np.ndarray(*,*,3) : image to convolve
    kernels : (np.ndarray(*,*)) or (np.ndarray(*,*,*)) : kernels used, aka filters
    res : (np.ndarray) = None : return array, the result is also returned like a classical function
    mode : "same" or "valid" or "full" = "same" : convolution padding
    vtype : (np type) = np.float32 : output type
    prefetching : (bool) : False if kernels and colors are stacked along the 3rd dimension
    ********* Examples *********
    fast_convolve(img, kernels)
    fast_convolve(img, wavelets.gabor2d(20, 3, 2, np.arange(6)/float(6)*np.pi, zeromean=True))
    fast_convolve(img, kernels, mode='valid')
    fast_convolve(img, kernels, mode='full')
    fast_convolve(img, kernels, mode='same', prefetching=False)
    fast_convolve(img, wavelets.gabor2d(20, 3, 2, 0), res=np.zeros(img.shape))
    """
    if len(kernels.shape) == 2:
        if res is None:
            if (mode == 'same'):
                res = np.zeros(img.shape + (1,), dtype=vtype)
            elif (mode == 'valid'):
                res = np.zeros((img.shape[0]-kernels.shape[0]+1, img.shape[1]-kernels.shape[1]+1, img.shape[2], 1), dtype=vtype)
            elif (mode == 'full'):
                res = np.zeros((img.shape[0]+kernels.shape[0]-1, img.shape[1]+kernels.shape[1]-1, img.shape[2], 1), dtype=vtype)
        else:
            res = np.stack([res], axis=3)
        test_scipy_fft(img, np.stack([kernels], axis=2), res, mode=mode, prefetching=prefetching)
        res = res[:,:,:,0]
    else:
        if res is None:
            if (mode == 'same'):
                res = np.zeros(img.shape + (kernels.shape[2],), dtype=vtype)
            elif (mode == 'valid'):
                res = np.zeros((img.shape[0]-kernels.shape[0]+1, img.shape[1]-kernels.shape[1]+1, img.shape[2], kernels.shape[2]), dtype=vtype)
            elif (mode == 'full'):
                res = np.zeros((img.shape[0]+kernels.shape[0]-1, img.shape[1]+kernels.shape[1]-1, img.shape[2], kernels.shape[2]), dtype=vtype)
        test_scipy_fft(img, kernels, res, mode=mode, prefetching=prefetching)
    return res
        
        



def test_scipy_fft(a, b, c, mode='same', prefetching=False):
    """
    """
    if prefetching:
        for i_n in np.arange(a.shape[0]):
            for i_m in np.arange(b.shape[0]):
                c[i_n, i_m, :, :] = np.real(fftconvolve(a[i_n, :, :], b[i_m, :, :], mode=mode))
    else:
        for i_n in np.arange(a.shape[-1]):
            for i_m in np.arange(b.shape[-1]):
                c[:, :, i_n, i_m] = np.real(fftconvolve(a[:, :, i_n], b[:, :, i_m], mode=mode))



   
def run_examples():
    """
    ********* Description *********
    For all functions defined in this file, it tests the examples in the function documentation"
    """
    mod = __file__[:-3]
    if mod[-1] == '.':
        mod = mod[:-1]
    exec("import " + mod + " as _" + mod)
    # Here we define variables that can be used in some examples"
    import wavelets
    from scipy import misc
    img = misc.face()
    kernels = wavelets.gabor2d(20, 3, 2, np.arange(6)/float(6)*np.pi, zeromean=True)
    # End of definition of those variables
    with open(mod + ".py", "r") as f:
        r = f.read()
    fu = " ".join([i.split("def ")[1].split("(")[0] for i in r.split('\n') if "def " in i])
    exec("fss = dir(_" + mod + ")")
    fs = [i for i in fss if i[:2] != '__' and i in fu]
    count_s = 0
    count_f = 0
    funcs = []
    for f in fs:
        try:
            exec("d = _" + mod + "." + f + ".func_doc")
            funcs.append(f)
        except AttributeError:
            pass
    for f in funcs:
        print "---------> " + f + " <---------"
        try:
            exec("d = _" + mod + "." + f + ".func_doc")
            es = [i.strip() for i in (d.split('********* Examples *********')[1]).split('\n')]
            for e in [i for i in es if i != '']:
                try:
                    ee = e
                    for ff in funcs:
                        ee = ee.replace(ff, "_" + mod + "." + ff)
                    exec(ee)
                    count_s += 1
                except Exception as ex:
                    count_f += 1
                    print "Example '" +e+ "' has failed with error :\n" + "\n".join(list(ex.args))+"\n"
                    pass
        except AttributeError:
            pass
        except IndexError:
            pass
    print "Number of examples ran : {}".format(count_f + count_s)
    print "Number of examples failed : {}".format(count_f)
    
 



# na = 6
# kernels = gabor2d(20, 3, 2, np.arange(na)/float(na)*np.pi, zeromean=True)
# #kernels = gabor2d(100, 10, 2, [0.3], zeromean=False)
# from scipy import misc
# img = misc.face()
# r = fast_convolve(img, kernels, mode='valid')
# show_wave(np.sum(np.abs(r), axis=3))
