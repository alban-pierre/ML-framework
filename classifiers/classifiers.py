# from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import time
import os
import sys
try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list, set, dict, np.ndarray)

from .includes import *

from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)



def help():
    print("""
Classifier, ie takes an array as input and outputs 0 or 1

Uses the package scikit-learn 

This link explains the possible classifiers choices :
    http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

Interesting as well :
    http://scikit-learn.org/stable/model_selection.html

Non exhaustive list of possible classifiers :
    KNeighborsClassifier(3)
    SVC(kernel="linear", C=0.025)
    SVC(kernel="sigmoid", C=0.025)
    SVC(kernel="rbf", C=0.025)
    SVC(gamma=2, C=1)
    GaussianProcessClassifier(1.0 * RBF(1.0))
    DecisionTreeClassifier(max_depth=5)
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    MLPClassifier(alpha=1)
    AdaBoostClassifier()
    GaussianNB()
    QuadraticDiscriminantAnalysis()
    LogisticRegression()
    LogisticRegression(solver="newton-cg")
    LogisticRegression(solver="lbfgs")
    LogisticRegression(solver="liblinear", penalty="l1")
    LogisticRegression(solver="sag")
    LogisticRegression(solver="saga", penalty="l1")
    LogisticRegressionCV()

Non exhaustive list of possible kernels :
    "linear"
    "sigmoid"
    "rbf"
    Linear()
    Polynomial()
    RBF()
    Cossim()
    Exponential()
    Laplacian()
    RationalQuadratic()
    InverseMultiquadratic()
    Cauchy()
    TStudent()
    ANOVA()
    Wavelet()
    Fourier()
    Tanimoto()
    Sorensen()
    AdditiveChi2()
    Chi2()
    Min()
    GeneralizedHistogramIntersection()
    MinMax()
    Spline()
    Log()
    Power()

Example of use :
    # Create data
    x_train = np.random.randn(1000,2)
    y_train = (np.mean(x_train, axis=1) > 0).astype(int)
    x_test = np.random.randn(1000,2)
    y_test = (np.mean(x_test, axis=1) > 0).astype(int)
    
    # Chose classifier
    clf = KNeighborsClassifier(3)
    
    # Fit the model
    clf.fit(x_train, y_train)
    
    # Compute and print score
    score_train = clf.score(x_train, y_train)
    score_test = clf.score(x_test, y_test)
    print "Train score : {}".format(score_train)
    print "Test score : {}".format(score_test)

    # Show classification
    show_classification(clf, x_test, y_test)
""")


    
def show_classification(clf, x, y):
    """
    ********* Description *********
    Print a 2D decision space
    ********* Params *********
    clf : (sklearn.classifier) : classifier
    x : np.ndarray(n, 2) : points
    y : np.ndarray(n, 1) : targets
    ********* Examples *********
    x, y = load_dataset()
    clf = KNeighborsClassifier(3)
    clf.fit(x, y)
    show_classification(clf, x, y)
    """
    if (len(x.shape) != 2):
        print("Warning : you cannot use the function show_classification if you are not in 2D")
    else:
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x_span = x_max-x_min
        x_min = x_min - x_span/20.
        x_max = x_max + x_span/20.
        h = x_span/100
        xx, yy = np.meshgrid(np.arange(x_min[0], x_max[0], h[0]), np.arange(x_min[1], x_max[1], h[1]))
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright, edgecolors='k')
        plt.show()



def get_classifiers_from(s):
    """
    Execute a string or file containing classifiers, in python 2 or 3
    It outputs the classifiers defined
    """
    if (s[-3:] == ".py"):
        with open(s) as f:
            r = f.read()
    else:
        r = s
    if (sys.version_info[0] == 2):
        exec(r)
        return classifiers
    else:
        namespace = globals()
        exec(r, namespace)
        return namespace['classifiers']


    
def get_classifiers(n=0):
    """
    ********* Description *********
    Return a list of classifiers, bigger or smaller depending of the value of n
    ********* Params *********
    n : (int) or (str) = 0 : instructions over the classifiers to return : 
     - if n is negative, it returns only one classifiern
     - if n is zero (default value), it returns one classifier of each type
     - if n is strictly positive, it returns more classifiers (Not implemented yet)
     - if n is filename, it executes the code inside and returns the contents of var "classifiers"
     - if n is a string, it executes n and returns the contents of variable "classifiers"
     - otherwise, it returns an empty list
    ********* Examples *********
    clfs = get_classifiers()
    clfs = get_classifiers(0)
    clfs = get_classifiers("clf_lists/one_of_each.py")
    clfs = get_classifiers("classifiers = [('a1', SVC(C=1)), ('a.5', SVC(C=0.5))]")
    """
    try:
        if isinstance(n, int):
            if (n < 0):
                classifiers = [("Nearest Neighbors", KNeighborsClassifier(3))]
            elif (n == 0):
                this_file_path = '/'.join(__file__.split('/')[:-1])
                filename = os.path.join(this_file_path, "clf_lists/one_of_each.py")
                classifiers = get_classifiers_from(filename)
            elif (n == 1):
                this_file_path = '/'.join(__file__.split('/')[:-1])
                filename = os.path.join(this_file_path, "clf_lists/one_of_each_plus_reg.py")
                classifiers = get_classifiers_from(filename)
                # with open(os.path.join(this_file_path, "clf_lists/one_of_each.py")) as f:
                #     r = f.read()
                #     classifiers = _execute_classifier_file(r)
            else:
                classifiers = []
        elif isinstance(n, str):
            classifiers = get_classifiers_from(n)
            # if (n[-3:] == ".py"):
            #     with open(n) as f:
            #         r = f.read()
            #         classifiers = _execute_classifier_file(r)
            # else:
            #     classifiers = _execute_classifier_file(n)
        else:
            classifiers = []
    except:
        print("Error while loading a list of classifiers, the error is likely to be in the argument n.")
        raise
    return [(i[0], i[1]) for i in classifiers]



def run_one_classifier(x_train, y_train, clf, x_test=None, y_test=None, verbose=True, show=True, i=""):
    """
    ********* Description *********
    Fit and return the score of one classifier
    ********* Params *********
    x_train : (np.ndarray(n, dx)) : points
    x_train : (np.ndarray(n, dy)) : targets
    clf : (sklearn.classifier) : classifier used
    x_test : np.ndarray(m, dx) or (int) or (float) = None : test points, or
        indication to use K-fold separation on x_train, 
        more precisely if (int) then the train is on (n-x_test) points, 
        and if (float) then the train is on (n*(1-x_test)) points
        if None we don't compute test score
    y_test : np.ndarray(m, dx) = None : test target
    verbose : (bool) = True : whether we print the classifier score
    show : (bool) = True : whether we plot the classifier
    i : (str) or (int) : the index of this classifier, generally used by run_all_classifiers
    ********* Return *********
    (score_train, score_test, running_time)
    (None, None, None) if it somehow failed
    ********* Examples *********
    x, y = load_dataset()
    clf = KNeighborsClassifier(3)
    score_train, score_test = run_one_classifier(x, y, clf)
    score_train, score_test = run_one_classifier(x, y, clf, show=False)
    score_train, score_test = run_one_classifier(x, y, clf, show=False, verbose=False)
    score_train, score_test = run_one_classifier(x, y, clf, i=666)
    """
    return _run_one_classifier(x_train, y_train, clf, x_test=x_test, y_test=y_test, verbose=verbose, show=show, i=i)


    
def _run_one_classifier(x_train, y_train, clf, x_test=None, y_test=None, verbose=True, show=True, i="", _score_test=None, _run_time=None):
    """
    Hidden function that is used by run_all_classifier
    _score_test : (float) = None : if x_test = None, we set the test score to this value
    _run_time : (float) = None : to overwrite the running time
    """
    # We define clf and name etc
    clf, name = _get_clf_attributes(clf)
    # We separate the train test data if asked of
    if isinstance(x_test, int):
        test_size = float(x_test)/x_train.shape[0]
        x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=test_size)
    elif isinstance(x_test, float):
        test_size=x_test
        x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=test_size)
    else:
        x_tr, x_te, y_tr, y_te = (x_train, x_test, y_train, y_test)
    # We run the classification
    try:
        start_time = time.time()
        clf.fit(x_tr, y_tr)
        if (x_te is None) or (y_te is None):
            score_train = clf.score(x_tr, y_tr)
            score_test = _score_test
        else:
            score_train = clf.score(x_tr, y_tr)
            score_test = clf.score(x_te, y_te)
            x_tr = np.concatenate([x_tr, x_te], axis=0)
            y_tr = np.concatenate([y_tr, y_te], axis=0)
        run_time = time.time() - start_time
        if (_run_time is not None):
            run_time = _run_time
        if show:
            t = _repr_show(i, name, score_train, score_test)
            plt.title(t)
            show_classification(clf, x_tr, y_tr)
        if verbose:
            t = _repr_verbose(i, name, score_train, score_test, run_time)
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
    return (score_train, score_test, run_time)



def _get_clf_attributes(clf):
    # Return clf and name for a dict or tuple classifier
    name = ""
    if isinstance(clf, tuple):
        if len(clf) == 1:
            cl = clf[0]
        elif len(clf) > 1:
            name = clf[0]
            cl = clf[1]
    elif isinstance(clf, dict):
        cl = clf["clf"]
        if "name" in clf.keys():
            name = clf["name"]
    else:
        cl = clf
    return cl, name
    


def _repr_show(i, name, score_train, score_test=None):
    # Representation for a plot title when we test a classifier
    t = ("" if (i == "") else "{}\n".format(i))
    t += ("" if (name == "") else ("name : {}\n".format(name)))
    t += "score_train : {0:.3f}".format(score_train)
    t += ("" if (score_test is None) else "\nscore_test : {0:.3f}".format(score_test))
    return t



def _repr_verbose(i, name, score_train, score_test=None, run_time=None):
    # Representation of a verbose line when we test a classifier
    t = ("" if (i == "") else "{} : ".format(i))
    t += "score_train : {0:.3f}".format(score_train)
    t += ("" if (score_test is None) else " : score_test : {0:.3f}".format(score_test))
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



def run_all_classifiers(x_train, y_train, clfs=0, x_test=None, y_test=None, selection_algo=None, verbose=True, show=False, final_verbose=range(10), final_show=False, sort_key=None):
    """
    ********* Description *********
    Try several different classifiers, and can show and verbose some of them
    ********* Params *********
    x_train : (np.ndarray(n, dx)) : points
    y_train : (np.ndarray(n, dy)) : targets
    clfs : (int) or (str) or [(sklearn.classifier)] : classifiers used, with get_classifiers syntax
    x_test : np.ndarray(m, dx) or (int) or (float) = None : test points, or
        indication to use K-fold separation on x_train, 
        more precisely if (int) then the train is on (n-x_test) points, 
        and if (float) then the train is on (n*(1-x_test)) points
        if None we don't compute test score
    y_test : np.ndarray(m, dx) = None : test target
    selection_algo : (MAB class) = None : Rules the run sequence of classifiers, cf multi_armed_bandit
    verbose : (bool) or [(bool)] or [(int)] = True : whether we print the classifiers score
    show : (bool) or [(bool)] or [(int)] = False : whether we plot the classifiers
    final_verbose : (bool) or [(bool)] or [(int)] = range(10) : same as verbose but for clf classement
    final_show : (bool) or [(bool)] or [(int)] = False : same as show but for clf classement
    sort_key : (lambda clf -> float) = lambda x:-x["score_test"] : key for classifiers final classment
    ********* Return *********
    score of classifiers tested
    ********* Examples *********
    x, y = load_dataset("moons")
    scores = run_all_classifiers(x, y)
    scores = run_all_classifiers(x, y, x_test=0.1)
    scores = run_all_classifiers(x, y, x_test=0.1, final_verbose=range(3))
    scores = run_all_classifiers(x, y, x_test=0.1, final_verbose=[True, True, True])
    scores = run_all_classifiers(x, y, x_test=0.1, verbose=False)
    sel = Uniform_MAB(1, 100) # Will run 100 tests
    scores = run_all_classifiers(x, y, x_test=0.1, verbose=True, selection_algo=sel)
    sel = Uniform_MAB(1, None, 8) # Will run during 8 seconds
    scores = run_all_classifiers(x, y, x_test=0.1, verbose=False, selection_algo=sel)
    """
    # We define sort_key
    if (sort_key is None):
        sort_key = lambda x: (-x["score_train"] if (x["score_test"] is None) else -x["score_test"])
    # We define clfs
    if isinstance(clfs, int) or isinstance(clfs, str):
        clfs = get_classifiers(clfs)
    # We properly define show, ie it will be a list of bool
    show = _verbose_show_proper(len(clfs), show)
    verbose = _verbose_show_proper(len(clfs), verbose)
    # We properly define test_size
    if isinstance(x_test, int):
        test_size = float(x_test)/x_train.shape[0]
    elif isinstance(x_test, float):
        test_size=x_test
    else:
        test_size=None
    # We run all the classifiers following selection_algo
    if any(verbose):
        print("\n\n")
    nbr_ex = 0
    start_time = time.time()
    if selection_algo is None:
        # In this section there are no particular clf selection
        # We separate the train test data if asked of
        if x_test is None:
            x_tr, x_te, y_tr, y_te = (x_train, x_test, y_train, y_test)
        else:
            x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=test_size)
        # We try over all classifiers
        scores = []
        for ic, sho, verb, clf in zip(range(len(show)), show, verbose, clfs):
            nbr_ex += 1
            tr, te, ti = _run_one_classifier(x_tr, y_tr, clf, x_te, y_te, verbose=verb, show=sho, i=ic)
            scores.append({"i":ic, "score_train":tr, "score_test":te, "clf":clf, "time":ti})
    else:
        # In this section we follow the class selection_algo to perform the classifiers tests
        selection_algo.set(n_arms=len(clfs))
        arm = selection_algo.next_arm()
        while (arm is not None):
            # We separate the train test data if asked of
            if x_test is None:
                x_tr, x_te, y_tr, y_te = (x_train, x_test, y_train, y_test)
            else:
                x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=test_size)
            tr, te, ti = _run_one_classifier(x_tr, y_tr, clfs[arm], x_te, y_te, verbose[arm], show[arm], i=nbr_ex)
            selection_algo.update_reward(te, arm=arm, other_data=(tr, ti))
            arm = selection_algo.next_arm()
            nbr_ex += 1
        scores = []
        for ic, tri, te, clf in zip(range(len(clfs)), selection_algo.other_data, selection_algo.mean_rewards, clfs):
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
            scores.append({"i":ic, "score_train":mean_err_tr, "score_test":te, "clf":clf, "time":mean_err_ti})
    # Now we have finished the tests of the classifiers
    # We print the final results obtained
    final_show = _verbose_show_proper(nbr_ex, final_show)
    final_verbose = _verbose_show_proper(nbr_ex, final_verbose)
    if any(verbose) or any(final_verbose):
        print("\nFinished running {} examples in {} seconds\n".format(nbr_ex, time.time() - start_time))
    if any(final_verbose) or any(final_show):
        scores_sorted = [e for e in scores if e["time"] is not None]
        scores_sorted = sorted(scores_sorted, key=sort_key)
        for ic, ss, sho, verb in zip(range(len(final_show)), scores_sorted, final_show, final_verbose):
            if verb or sho:
                _run_one_classifier(x_train, y_train, ss["clf"], verbose=verb, show=sho, i=ic, _score_test=ss["score_test"], _run_time=ss["time"])
    return scores



def load_dataset(dataset="moons"):
    """
    ********* Description *********
    Load simple common datasets
    ********* Params *********
    dataset : (str) : "moons" : the dataset name to use
    ********* Return *********
    (x, y) : points, targets
    ********* Examples *********
    x, y = load_dataset(dataset="moons")
    x, y = load_dataset(dataset="circles")
    """
    if (dataset == "moons"):
        from sklearn.datasets import make_moons
        return make_moons(noise=0.3, random_state=0)
    elif (dataset == "circles"):
        from sklearn.datasets import make_circles
        return make_circles(noise=0.2, factor=0.5, random_state=1)
    

        
def run_examples(verbose=True, show=True, dataset="moons"):
    """
    ********* Description *********
    Run some classifiers and show them as an example
    ********* Params *********
    verbose : (bool) = True : whether we print the classifier score
    show : (bool) = True : whether we plot the classifier score
    dataset : (str) = "moons" : which dataset to use
    ********* Examples *********
    run_examples()
    run_examples(show=False)
    run_examples(verbose=False, show=False) # You will get nothing from that !
    """
    x, y = load_dataset(dataset)
    run_all_classifiers(x, y, clfs=0, verbose=verbose, show=show, x_test=0.1)
