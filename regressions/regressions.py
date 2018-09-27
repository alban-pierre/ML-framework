import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import os
import sys
import time
try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list, set, dict, np.ndarray)
    
# from pykernels.pykernels.basic import *

from .includes import *

from sklearn.exceptions import DataConversionWarning

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)




def help():
    print("""
Regressions, ie takes an array as input and outputs a float

Uses the package scikit-learn 

This link explains the possible regressions choices :
    http://scikit-learn.org/stable/modules/linear_model.html

Interesting as well :
    http://scikit-learn.org/stable/model_selection.html

Non exhaustive list of possible regressions :
    LinearRegression()
    Ridge(alpha = .5)
    RidgeCV(alphas=[0.1, 1.0, 10.0])
    Lasso(alpha = 0.1)
    LassoCV(alphas=[0.1, 1.0, 10.0])
    LassoLars(alpha = 0.1)
    LassoLarsCV()
    LassoLarsIC()
    ElasticNet(alpha=1.0, l1_ratio=0.5)
    ElasticNetCV(alphas=[0.1, 1.0, 10.0], l1_ratio=[0.1, 0.5, 0.9])
    Lars()
    LarsCV()
    OrthogonalMatchingPursuit(n_nonzero_coefs = 1)
    BayesianRidge()
    ARDRegression()
    SGDRegressor()
    PassiveAggressiveRegressor()
    RANSACRegressor()
    TheilSenRegressor()
    HuberRegressor()
    KernelRidge()
    GaussianProcessRegressor()
    SVR()

Example of use :
    # Create data
    x_train = np.random.randn(1000,2)
    y_train = np.dot(x_train, [2,5]) + np.random.randn(1000)/10.
    x_test = np.random.randn(1000,2)
    y_test = np.dot(x_test, [2,5]) + np.random.randn(1000)/10.
    # Data more plottable here :
    x_train = np.random.randn(1000,1)
    y_train = x_train*3 + 2 + np.random.randn(1000,1)/10.
    x_test = np.random.randn(1000,1)
    y_test = x_test*3 + 2 + np.random.randn(1000,1)/10.
    
    # Chose regression algorithm
    reg = LinearRegression()

    # Fit the model
    reg.fit(x_train, y_train)

    # Make prediction
    pred_y_train = reg.predict(x_train)
    pred_y_test = reg.predict(x_test)
    
    # Compute and print mean square error
    ms_err_train = mean_squared_error(pred_y_train, y_train)
    ms_err_test = mean_squared_error(pred_y_test, y_test)
    print "Train mean square error : {}".format(ms_err_train)
    print "Test mean square error : {}".format(ms_err_test)

    # Show regression
    show_regression(reg, x_test, y_test)
""")


    
def show_regression(reg, x, y):
    """
    ********* Description *********
    Plot a 2D regression with points
    ********* Params *********
    reg : (sklearn.regression) : regression
    x : np.ndarray(n, 1) : points
    y : np.ndarray(n, 1) : targets
    ********* Examples *********
    x, y = load_dataset()
    reg = Ridge()
    reg.fit(x, y)
    show_regression(reg, x, y)
    """
    if (len(x.shape) > 1) and (x.shape[1] != 1):
        print("Warning : you cannot use the function show_regression if you are not in 1D")
    else:
        if (len(x.shape) == 1):
            x_min = np.min(x)
            x_max = np.max(x)
        else:
            x_min = np.min(x[:,0])
            x_max = np.max(x[:,0])
        x_span = x_max-x_min
        x_min = x_min - x_span/20.
        x_max = x_max + x_span/20.
        h = x_span/100.
        xl = np.arange(x_min, x_max, h).reshape(-1, 1)
        yl = reg.predict(xl)
        plt.plot(x, y, '.k')
        plt.plot(xl, yl, 'r')
        plt.show()



def _execute_regression_file(s):
    """
    Execute a file containing regressions, in python 2 or 3
    It outputs the regressions defined
    """
    if (sys.version_info[0] == 2):
        exec(s)
        return regressions
    else:
        namespace = globals()
        exec(s, namespace)
        return namespace['regressions']
    


def get_regressions(n=0):
    """
    ********* Description *********
    Return a list of regressions, bigger or smaller depending of the value of n
    ********* Params *********
    n : (int) or (str) = 0 : instructions over the regressions to return : 
     - if n is negative, it returns only one regression
     - if n is zero (default value), it returns one regression of each type
     - if n is strictly positive, it returns more regressions (Not implemented yet)
     - if n is filename, it executes the code inside and returns the contents of variable "regressions"
     - if n is a string, it executes n and returns the contents of variable "regressions"
     - otherwise, it returns an empty list
    ********* Examples *********
    regs = get_regressions()
    regs = get_regressions(0)
    regs = get_regressions("reg_lists/one_of_each.py")
    regs = get_regressions("regressions = [('a.1', Ridge(alpha = .1)), ('a.5', Ridge(alpha = .5))]")
    """
    try:
        if isinstance(n, int):
            if (n < 0):
                regressions = [("Linear Regression", LinearRegression())]
            elif (n == 0):
                this_file_path = '/'.join(__file__.split('/')[:-1])
                with open(os.path.join(this_file_path, "reg_lists/one_of_each.py")) as f:
                    r = f.read()
                    regressions = _execute_regression_file(r)
            else:
                regressions = []
        elif isinstance(n, str):
            if (n[-3:] == ".py"):
                with open(n) as f:
                    r = f.read()
                    regressions = _execute_regression_file(r)
            else:
                regressions = _execute_regression_file(n)
        else:
            regressions = []
    except:
        print("Error while loading a list of regressions, the error is likely to be in the argument n")
        raise
    return [(i[0], i[1]) for i in regressions]



def run_one_regression(x_train, y_train, reg, error_func=mean_squared_error, x_test=None, y_test=None, verbose=True, show=True, i="", _error_test=None, _run_time=None):
    """
    ********* Description *********
    Fit and return the error of one regression
    ********* Params *********
    x_train : (np.ndarray(n, dx)) : points
    x_train : (np.ndarray(n, dy)) : targets
    reg : (sklearn.regression) : regression used
    error_func : (func) = sklearn.mean_squared_error : the error used
    x_test : np.ndarray(m, dx) or (int) or (float) = None : test points, or
        indication to use K-fold separation on x_train, 
        more precisely if (int) then the train is on (n-x_test) points, 
        and if (float) then the train is on (n*(1-x_test)) points
        if None we don't compute test error
    y_test : np.ndarray(m, dx) = None : test target
    verbose : (bool) = True : whether we print the regression error
    show : (bool) = True : whether we plot the regression
    i : (str) or (int) : the index of this regression, generally used by run_all_regressions
    ********* Return *********
    (error_train, error_test, running_time)
    (None, None, None) if it somehow failed
    ********* Examples *********
    x, y = load_dataset()
    reg = Ridge()
    error_train, error_test, run_time = run_one_regression(x, y, reg)
    error_train, error_test, run_time = run_one_regression(x, y, reg, show=False)
    error_train, error_test, run_time = run_one_regression(x, y, reg, show=False, verbose=False)
    error_train, error_test, run_time = run_one_regression(x, y, reg, i=666)
    """
    return _run_one_regression(x_train, y_train, reg, error_func=error_func, x_test=x_test, y_test=y_test, verbose=verbose, show=show, i=i)



def _run_one_regression(x_train, y_train, reg, error_func=mean_squared_error, x_test=None, y_test=None, verbose=True, show=True, i="", _error_test=None, _run_time=None):
    """
    Hidden function that is used by run_all_regression
    _error_test : (float) = None : if x_test = None, we set the test error to this value
    _run_time : (float) = None : to overwrite the running time
    """
    # We define reg and name etc
    reg, name = _get_reg_attributes(reg)
    # We run the regression
    try:
        start_time = time.time()
        reg.fit(x_train, y_train)
        if (x_test is None) or (y_test is None):
            error_train = error_func(reg.predict(x_train), y_train)
            error_test = _error_test
        else:
            error_train = error_func(reg.predict(x_train), y_train)
            error_test = error_func(reg.predict(x_test), y_test)
            x_train = np.concatenate([x_train, x_test], axis=0)
            y_train = np.concatenate([y_train, y_test], axis=0)
        run_time = time.time() - start_time
        if (_run_time is not None):
            run_time = _run_time
        if show:
            t = _repr_show(i, name, error_train, error_test)
            plt.title(t)
            show_regression(reg, x_train, y_train)
        if verbose:
            t = _repr_verbose(i, name, error_train, error_test, run_time)
            print(t)
    except ValueError:
        print("Kernel failed with the data provided : {}".format(name))
        return (None, None, None)
    except AttributeError:
        return (None, None, None)
    except KeyboardInterrupt:
        raise
    except:
        print("Kernel failed : {}".format(name))
        return (None, None, None)
    return (error_train, error_test, run_time)



def _get_reg_attributes(reg):
    # Return reg and name for a dict or tuple regression
    name = ""
    if isinstance(reg, tuple):
        if len(reg) == 1:
            rg = reg[0]
        elif len(reg) > 1:
            name = reg[0]
            rg = reg[1]
    elif isinstance(reg, dict):
        rg = reg["reg"]
        if "name" in reg.keys():
            name = reg["name"]
    else:
        rg = reg
    return rg, name
    


def _repr_show(i, name, error_train, error_test=None):
    # Representation for a plot title when we test a regression
    t = ("" if (i == "") else "{}\n".format(i))
    t += ("" if (name == "") else ("name : {}\n".format(name)))
    t += "error_train : {0:.3f}".format(error_train)
    t += ("" if (error_test is None) else "\nerror_test : {0:.3f}".format(error_test))
    return t



def _repr_verbose(i, name, error_train, error_test=None, run_time=None):
    # Representation of a verbose line when we test a regression
    t = ("" if (i == "") else "{} : ".format(i))
    t += "error_train : {0:.3f}".format(error_train)
    t += ("" if (error_test is None) else " : error_test : {0:.3f}".format(error_test))
    t += ("" if (run_time is None) else " : run_time : {0:.3f}".format(run_time))
    t += ("" if (name == "") else ("   -   name : {}".format(name)))
    return t
            


def _verbose_show_proper(length, verbshow):
    # We properly define verbose (or show), ie it will be a list of bool
    if isinstance(verbshow, bool):
        res = [verbshow for i in range(length)]
    elif isinstance(verbshow, Iterable):
        if verbshow and isinstance(verbshow[0], bool):
            res = [False for i in range(length)]
            for i,j in enumerate(verbshow):
                if (i < len(res)):
                    res[i] = j
        else:
            res = [False for i in range(length)]
            for i in verbshow:
                if (i < len(res)):
                    res[i] = True
    else:
        res = [False for i in range(length)]
    return res



def run_all_regressions(x_train, y_train, regs=0, error_func=mean_squared_error, x_test=None, y_test=None, selection_algo=None, verbose=True, show=False, final_verbose=range(10), final_show=False, sort_key=None):
    """
    ********* Description *********
    Try several different regressions, and can show and verbose some of them
    ********* Params *********
    x_train : (np.ndarray(n, dx)) : points
    x_train : (np.ndarray(n, dy)) : targets
    regs : (int) or (str) or [(sklearn.regression)] : regressions used, with get_regressions syntax
    error_func : (func) = sklearn.mean_squared_error : the error used
    x_test : np.ndarray(m, dx) or (int) or (float) = None : test points, or
        indication to use K-fold separation on x_train, 
        more precisely if (int) then the train is on (n-x_test) points, 
        and if (float) then the train is on (n*(1-x_test)) points
        if None we don't compute test error
    y_test : np.ndarray(m, dx) = None : test target
    selection_algo : (MAB class) = None : Rules the run sequence of regressions, cf multi_armed_bandit
    verbose : (bool) or [(bool)] or [(int)] = True : whether we print the regressions error
    show : (bool) or [(bool)] or [(int)] = False : whether we plot the regressions
    final_verbose : (bool) or [(bool)] or [(int)] = range(10) : same as verbose but for reg classement
    final_show : (bool) or [(bool)] or [(int)] = False : same as show but for reg classement
    sort_key : (lambda reg -> float) = lambda x:x["error_test"] : key for regressions final classment
    ********* Return *********
    error of regressions tested
    ********* Examples *********
    x, y = load_dataset("boston")
    errors = run_all_regressions(x, y)
    errors = run_all_regressions(x, y, x_test=0.1)
    errors = run_all_regressions(x, y, x_test=0.1, final_verbose=range(3))
    errors = run_all_regressions(x, y, x_test=0.1, final_verbose=[True, True, True])
    errors = run_all_regressions(x, y, x_test=0.1, verbose=False)
    sel = Uniform_MAB(1, 100) # Will run 100 tests
    errors = run_all_regressions(x, y, x_test=0.1, verbose=True, selection_algo=sel)
    sel = Uniform_MAB(1, None, 8) # Will run during 8 seconds
    errors = run_all_regressions(x, y, x_test=0.1, verbose=False, selection_algo=sel)
    """
    # We define sort_key
    if (sort_key is None):
        sort_key = lambda x: (x["error_train"] if (x["error_test"] is None) else x["error_test"])
    # We define regs
    if isinstance(regs, int):
        regs = get_regressions(0)
    # We properly define show, ie it will be a list of bool
    show = _verbose_show_proper(len(regs), show)
    verbose = _verbose_show_proper(len(regs), verbose)
    # We properly define test_size
    if isinstance(x_test, int):
        test_size = float(x_test)/x_train.shape[0]
    elif isinstance(x_test, float):
        test_size=x_test
    else:
        test_size=None
    # We run all the regressions following selection_algo
    if any(verbose):
        print("\n\n")
    nbr_ex = 0
    start_time = time.time()
    if selection_algo is None:
        # In this section there are no particular reg selection
        # We separate the train test data if asked of
        if x_test is None:
            x_tr, x_te, y_tr, y_te = (x_train, x_test, y_train, y_test)
        else:
            x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=test_size)
        # We try over all regressions
        errors = []
        for ic, sho, verb, reg in zip(range(len(show)), show, verbose, regs):
            nbr_ex += 1
            tr, te, ti = run_one_regression(x_tr, y_tr, reg, error_func, x_te, y_te, verbose=verb, show=sho, i=ic)
            errors.append({"i":ic, "error_train":tr, "error_test":te, "reg":reg, "time":ti})
    else:
        # In this section we follow the class selection_algo to perform the regressions tests
        selection_algo.set(n_arms=len(regs))
        arm = selection_algo.next_arm()
        while (arm is not None):
            # We separate the train test data if asked of
            if x_test is None:
                x_tr, x_te, y_tr, y_te = (x_train, x_test, y_train, y_test)
            else:
                x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=test_size)
            tr, te, ti = run_one_regression(x_tr, y_tr, regs[arm], error_func, x_te, y_te, verbose[arm], show[arm], i=nbr_ex)
            selection_algo.update_reward(te, arm=arm, other_data=(tr, ti))
            arm = selection_algo.next_arm()
            nbr_ex += 1
        errors = []
        for ic, tri, te, reg in zip(range(len(regs)), selection_algo.other_data, selection_algo.mean_rewards, regs):
            # If an experiment failed, we set the mean to None
            tr = [j[0] for j in tri]
            if tr and (None not in tr):
                mean_err_tr = np.mean(tr)
            else:
                mean_err_tr = None
            ti = [j[1] for j in tri]
            if ti and (None not in ti):
                mean_err_ti = np.mean(ti)
            else:
                mean_err_ti = None
            errors.append({"i":ic, "error_train":mean_err_tr, "error_test":te, "reg":reg, "time":mean_err_ti})
    # Now we have finished the tests of the regressions
    # We print the final results obtained
    final_show = _verbose_show_proper(nbr_ex, final_show)
    final_verbose = _verbose_show_proper(nbr_ex, final_verbose)
    if any(verbose) or any(final_verbose):
        print("\nFinished running {} examples in {} seconds\n".format(nbr_ex, time.time() - start_time))
    if any(final_verbose) or any(final_show):
        errors_sorted = [e for e in errors if e["time"] is not None]
        errors_sorted = sorted(errors_sorted, key=sort_key)
        for ic, ss, sho, verb in zip(range(len(final_show)), errors_sorted, final_show, final_verbose):
            _run_one_regression(x_train, y_train, ss["reg"], error_func, verbose=verb, show=sho, i=ic, _error_test=ss["error_test"], _run_time=ss["time"])
    return errors



def load_dataset(dataset="default"):
    """
    ********* Description *********
    Load simple common datasets
    ********* Params *********
    dataset : (str) : "default" : the dataset name to use
    ********* Return *********
    (x, y) : points, targets
    ********* Examples *********
    x, y = load_dataset(dataset="default")
    x, y = load_dataset(dataset="boston")
    """
    if (dataset == "boston"):
        from sklearn.datasets import load_boston
        boston = load_boston()
        return preprocessing.scale(boston.data), preprocessing.scale(boston.target)
    x = np.random.randn(1000,1)
    y = x*3 + 2 + np.random.randn(1000,1)/10.
    return x, y
    

        
def run_examples(verbose=True, show=True, dataset="default"):
    """
    ********* Description *********
    Run some regressions and show them as an example
    ********* Params *********
    verbose : (bool) = True : whether we print the regression error
    show : (bool) = True : whether we plot the regression error
    dataset : (str) = "default" : which dataset to use
    ********* Examples *********
    run_examples()
    run_examples(show=False)
    run_examples(verbose=False, show=False) # You will get nothing from that !
    """
    x, y = load_dataset(dataset)
    run_all_regressions(x, y, regs=0, verbose=verbose, show=show, x_test=0.1)
