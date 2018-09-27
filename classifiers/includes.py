# try:
#     from pykernels.pykernels.basic import *
#     from pykernels.pykernels.regular import *
# except (ZeroDivisionError, ImportError):
#     from pykernels.basic import *
#     from pykernels.regular import *



from .pykernels.pykernels.basic import *
from .pykernels.pykernels.regular import *


class Empty_Classifier:
    """
    Class for defining classifications in case they can't be imported
    """
    def __init__(self, *args, **kargs):
        pass


try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError:
    print("Some classifiers packages could not be loaded, running failsafe mode")
    
    try:
        from sklearn.neural_network import MLPClassifier
    except ImportError as e:
        print(str(e))
        MLPClassifier = Empty_Classifier

    try:
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError as e:
        print(str(e))
        KNeighborsClassifier = Empty_Classifier

    try:
        from sklearn.svm import SVC
    except ImportError as e:
        print(str(e))
        SVC = Empty_Classifier

    try:
        from sklearn.svm import LinearSVC
    except ImportError as e:
        print(str(e))
        LinearSVC = Empty_Classifier

    try:
        from sklearn.linear_model import SGDClassifier
    except ImportError as e:
        print(str(e))
        SGDClassifier = Empty_Classifier

    try:
        from sklearn.gaussian_process import GaussianProcessClassifier
    except ImportError as e:
        print(str(e))
        GaussianProcessClassifier = Empty_Classifier

    try:
        from sklearn.gaussian_process.kernels import RBF
    except ImportError as e:
        print(str(e))
        RBF = Empty_Classifier

    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as e:
        print(str(e))
        DecisionTreeClassifier = Empty_Classifier

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        print(str(e))
        RandomForestClassifier = Empty_Classifier

    try:
        from sklearn.ensemble import AdaBoostClassifier
    except ImportError as e:
        print(str(e))
        AdaBoostClassifier = Empty_Classifier

    try:
        from sklearn.naive_bayes import GaussianNB
    except ImportError as e:
        print(str(e))
        GaussianNB = Empty_Classifier

    try:
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    except ImportError as e:
        print(str(e))
        QuadraticDiscriminantAnalysis = Empty_Classifier

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        print(str(e))
        LogisticRegression = Empty_Classifier

    try:
        from sklearn.linear_model import LogisticRegressionCV
    except ImportError as e:
        print(str(e))
        LogisticRegressionCV = Empty_Classifier

    try:
        from sklearn.linear_model import Perceptron
    except ImportError as e:
        print(str(e))
        Perceptron = Empty_Classifier

    try:
        from sklearn.linear_model import PassiveAggressiveClassifier
    except ImportError as e:
        print(str(e))
        PassiveAggressiveClassifier = Empty_Classifier

    try:
        from sklearn.linear_model import RidgeClassifier
    except ImportError as e:
        print(str(e))
        RidgeClassifier = Empty_Classifier

    try:
        from sklearn.linear_model import RidgeClassifierCV
    except ImportError as e:
        print(str(e))
        RidgeClassifierCV = Empty_Classifier

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        print(str(e))
        RandomForestClassifier = Empty_Classifier

    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as e:
        print(str(e))
        ExtraTreesClassifier = Empty_Classifier

    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError as e:
        print(str(e))
        GradientBoostingClassifier = Empty_Classifier
         
