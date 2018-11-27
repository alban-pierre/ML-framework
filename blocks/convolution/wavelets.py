import numpy as np
#from scipy import signal
import matplotlib.pyplot as plt



def to_rgb_image(img):
    """
    ********* Description *********
    Shifts 3D arrays to make an RGB image (0-255)(uint8)
    ********* Params *********
    img : np.ndarray(*,*,3) : The image
    ********* Examples *********
    #to_rgb_image(img)
    """
    mx = np.max(img)
    mn = np.min(img)
    return ((img - mn)/((mx-mn)/255)).astype(np.uint8)



def show_wave(wave, cmap="gray"):
    """
    ********* Description *********
    Show 1D or 2D wave in matplotlib
    ********* Params *********
    wave : (np.ndarray) : the wave to plot
    ********* Examples *********
    show_wave(wave)
    show_wave(gabor1d(100, 10, 2))
    show_wave(gaussian2d(100, 10))
    show_wave(gaussian2d(100, 10), cmap="jet")
    show_wave(gabor2d(100, 20, 2, 0.3), cmap="jet")
    """
    if len(wave.shape) == 1: # 1D data, lines
        if (type(wave[0]) == np.complex64) or (type(wave[0]) == np.complex128):
            plt.plot(np.abs(wave), color="black")
            plt.plot(np.real(wave), color="blue")
            plt.plot(np.imag(wave), color="red")
            plt.title("black=abs, blue=real, red=imag")
            plt.show()
        else:
            plt.plot(wave, color="black")
            plt.show()
    if len(wave.shape) == 2: # 2D data, grey images
        if (type(wave[0][0]) == np.complex64) or (type(wave[0][0]) == np.complex128):
            plt.imshow(np.abs(wave), cmap=cmap)
            plt.title("abs")
            plt.show()
            plt.imshow(np.real(wave), cmap=cmap)
            plt.title("real")
            plt.show()
            plt.imshow(np.imag(wave), cmap=cmap)
            plt.title("imag")
            plt.show()
        else:
            plt.imshow(wave, cmap=cmap)
            plt.show()
    if len(wave.shape) == 3:
        if wave.shape[2] == 3: # 2D data, rgb images
            if (type(wave[0][0][0]) == np.complex64) or (type(wave[0][0][0]) == np.complex128):
                plt.imshow(to_rgb_image(np.abs(wave)))
                plt.title("abs")
                plt.show()
                plt.imshow(to_rgb_image(np.real(wave)))
                plt.title("real")
                plt.show()
                plt.imshow(to_rgb_image(np.imag(wave)))
                plt.title("imag")
                plt.show()
            else:
                plt.imshow(to_rgb_image(wave))
                plt.show()
                

                
def gaussian1d(shape, std, x0="center", vtype=np.float32):
    """
    ********* Description *********
    Return a one dimensionnal gaussian
    ********* Params *********
    shape : (int) : return size
    std : (float) or [(float)] : standard deviation(s)
    x0 : (int) or (str) = "center" : gaussian center
    vtype : (np type) = np.float32 : output type
    ********* Return *********
    x = ((0:shape)-x0)/std
    return exp(-x**2)
    ********* Examples *********
    gaussian1d(100, 10)
    gaussian1d(100, 20)
    gaussian1d(100, 10, 0)
    gaussian1d(100, [10, 20, 30])
    gaussian1d(100, 10, vtype=np.float64)
    """
    if type(x0) == str:
        if x0 == "center":
            x0 = shape/2.-.5
    if (type(std) != list) and (type(std) != np.ndarray) and (type(std) != tuple):
        return np.exp(-((np.arange(shape).astype(vtype) - x0)/std)**2).astype(vtype)
    else:
        return np.stack([np.exp(-((np.arange(shape).astype(vtype) - x0)/sigma)**2).astype(vtype) for sigma in std], axis=1)



          

def gaussian2d(shape, std, x0="center", vtype=np.float32):
    """
    ********* Description *********
    Return a two dimensionnal gaussian
    ********* Params *********
    shape : (int) or ((int),(int)) : return size
    std : (float) or ((float),(float)) or [(float) or ((float),(float))]: standard deviation(s)
    x0 : (int) or ((int),(int)) or (str) = "center" : gaussian center
    vtype : (np type) = np.float32 : output type
    ********* Return *********
    x = ((0:shape)-x0)/std
    return exp(-x**2)*exp(-y**2)
    ********* Examples *********
    gaussian2d(100, 10)
    gaussian2d(100, 20)
    gaussian2d(100, 10, 0)
    gaussian2d((50,100), (5,10))
    gaussian2d((50,100), (5,10), 10)
    gaussian2d((50,100), (5,10), vtype=np.float64)
    gaussian2d((100,100), [30,(10,50),(50,10)])
    """
    if type(shape) != tuple:
        shape = (shape, shape)
    if type(x0) != tuple:
        x0 = (x0, x0)
    if (type(std) != list) and (type(std) != np.ndarray) and (type(std) != tuple):
        std = (std, std)
    elif (type(std) != tuple) or (len(std) != 2):
        std = [sigma if (type(sigma) == tuple) else (sigma, sigma) for sigma in std]
        return np.stack([np.outer(gaussian1d(shape[0], sigma[0], x0[0], vtype), gaussian1d(shape[1], sigma[1], x0[1], vtype)) for sigma in std], axis=2)
    y = gaussian1d(shape[0], std[0], x0[0], vtype)
    x = gaussian1d(shape[1], std[1], x0[1], vtype)
    return np.outer(y, x)





def gabor1d(shape, std, w, w0=0, x0="center", zeromean=False, vtype=np.complex64):
    """
    ********* Description *********
    Return a one dimensionnal gabor wavelet
    ********* Params *********
    shape : (int) : wavelet size
    std : (float) or [(float)]: standard deviation(s)
    w : (float) : wave length
    w0 : (float) = 0 : wave complex shift
    x0 : (int) or (str) = "center" : wavelet center
    zeromean : (bool) = False : if the wavelet sum to zero
    vtype : (np type) = np.complex64 : output type
    ********* Return *********
    x = ((0:shape)-x0)/std
    return exp(-x**2)*exp(-i(wx-w0))
    ********* Examples *********
    gabor1d(100, 10, 2)
    gabor1d(100, 20, 2)
    gabor1d(100, 20, 7)
    gabor1d(100, 10, 2, np.pi)
    gabor1d(100, 10, 2, np.pi, 0)
    gabor1d(100, 10, 2, np.pi, "center", vtype=np.complex128)
    gabor1d(100, 10, 2, np.pi, "center", True)
    gabor1d(100, [10, 20, 30], 2)
    """
    if type(x0) == str:
        if x0 == "center":
            x0 = shape/2.-.5
    if (type(std) != list) and (type(std) != np.ndarray) and (type(std) != tuple):
        if (vtype == np.complex128) or (vtype == np.float64):
            x = (np.arange(shape).astype(np.float64) - x0)/std
        else:
            x = (np.arange(shape).astype(np.float32) - x0)/std
        if zeromean:
            g = np.exp(-x**2).astype(vtype)
            s = np.exp(-1j*w*x+1j*w0).astype(vtype)
            m = np.mean(g*s)/np.mean(g)
            return (g*(s-m))
        else:
            return np.exp(-x**2-1j*w*x+1j*w0).astype(vtype)
    else:
        return np.stack([gabor1d(shape, sigma, w, w0, x0, zeromean, vtype) for sigma in std], axis=1)

    

def gabor2d(shape, std, w, theta, w0=0, x0="center", zeromean=False, vtype=np.complex64):
    """
    ********* Description *********
    Return a two dimensionnal gabor wavelet
    ********* Params *********
    shape : (int) or ((int),(int)) : wavelet size
    std : (float) or ((float),(float)) or [(float) or ((float),(float))]: standard deviation(s)
    w : (float) : wave length
    theta : (float) or [(float)]: wave orientation(s)
    w0 : (float) = 0 : wave complex shift
    x0 : (int) or ((int),(int)) or (str) = "center" : wave center
    vtype : (np type) = np.complex64 : output type
    ********* Return *********
    x = ((0:shape)-x0)/std
    return exp(-x**2)*exp(-i(wx-w0))
    ********* Examples *********
    gabor2d(100, 10, 2, 0)
    gabor2d(100, 20, 2, 0)
    gabor2d(100, (10, 20), 2, 0)
    gabor2d(100, 20, 2, 0.3)
    gabor2d(100, 20, 7, 0.3)
    gabor2d(100, 20, 2, [0, 0.3, 0.6])
    gabor2d(100, 10, 2, 0.3, np.pi)
    gabor2d(100, 10, 2, 0.3, np.pi, 0)
    gabor2d((50,100), 10, 2, 0.3, np.pi, "center")
    gabor2d(100, 10, 2, 0.3, np.pi, "center", vtype=np.complex128)
    gabor2d(100, 10, 2, 0.3, np.pi, "center", True)
    gabor2d(100, 10, 2, [0, 0.3, 0.6], np.pi, "center", True)
    gabor2d(100, [5, 10, 20], 2, 0, zeromean=True)
    gabor2d(100, [5, 10, 20], 2, [0, 0.3, 0.6], zeromean=True)
    """
    if type(shape) != tuple:
        shape = (shape, shape)
    # if type(std) != tuple:
    #     std = (std, std)
    if type(x0) != tuple:
        x0 = (x0, x0)
    # if (type(theta) != list) and (type(theta) != np.ndarray) and (type(theta) != tuple):
    #     theta = [theta]
    if type(x0[0]) == str:
        if x0[0] == "center":
            x0 = (shape[0]/2.-.5, x0[1])
    if type(x0[1]) == str:
        if x0[1] == "center":
            x0 = (x0[0], shape[1]/2.-.5)
    if (type(std) != list) and (type(std) != np.ndarray) and (type(std) != tuple):
        std = (std, std)
    elif (type(std) != tuple) or (len(std) != 2):
        std = [sigma if (type(sigma) == tuple) else (sigma, sigma) for sigma in std]
        if (type(theta) != list) and (type(theta) != np.ndarray) and (type(theta) != tuple):
            return np.stack([gabor2d(shape, sigma, w, theta, w0, x0, zeromean, vtype) for sigma in std], axis=2)
        else:
            return np.concatenate([gabor2d(shape, sigma, w, theta, w0, x0, zeromean, vtype) for sigma in std], axis=2)
    if (vtype == np.complex128) or (vtype == np.float64):
        y = (np.arange(shape[0]).astype(np.float64) - x0[0])/std[0]
        x = (np.arange(shape[1]).astype(np.float64) - x0[1])/std[1]
    else:
        y = (np.arange(shape[0]).astype(np.float32) - x0[0])/std[0]
        x = (np.arange(shape[1]).astype(np.float32) - x0[1])/std[1]
    g = np.outer(np.exp(-y**2), np.exp(-x**2)).astype(vtype)
    if zeromean:
        # if len(theta) == 1:
        if (type(theta) != list) and (type(theta) != np.ndarray) and (type(theta) != tuple):
            #s = np.exp(-1j*w*np.add.outer(np.sin(theta[0])*y,-np.cos(theta[0])*x)+1j*w0).astype(vtype)
            s = np.exp(-1j*w*np.add.outer(np.sin(theta)*y,-np.cos(theta)*x)+1j*w0).astype(vtype)
            m = np.mean(g*s)/np.mean(g)
            return (g*(s-m))
        else:
            mg = np.mean(g)
            return np.stack([(g*(s-m))  for (s,m) in [(s, np.mean(g*s)/mg) for s in [np.exp(-1j*w*np.add.outer(np.sin(t)*y,-np.cos(t)*x)+1j*w0).astype(vtype) for t in theta]]], axis=2)
    else:
        # if len(theta) == 1:
        if (type(theta) != list) and (type(theta) != np.ndarray) and (type(theta) != tuple):
            #s = np.exp(-1j*w*np.add.outer(np.sin(theta[0])*y,-np.cos(theta[0])*x)+1j*w0).astype(vtype)
            s = np.exp(-1j*w*np.add.outer(np.sin(theta)*y,-np.cos(theta)*x)+1j*w0).astype(vtype)
            return g*s
        else:
            return np.stack([(g*np.exp(-1j*w*np.add.outer(np.sin(t)*y,-np.cos(t)*x)+1j*w0)).astype(vtype) for t in theta], axis=2)



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
    exec("wave = _" + mod + ".gaussian1d(100, 25)")
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
    
