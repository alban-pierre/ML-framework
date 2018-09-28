


class Regression_With_Custom_Kernel:
    """
    Class for syntax convenience when defining custom kernels
    """
    def __init__(self, reg, kernel):
        """
        reg : the scikit learn regression
        kernel : the custom kernel to use
        """
        self.reg = reg
        self.kernel = kernel
        self.reg.set_params(kernel="precomputed")
        
    def fit(self, x, y, *args, **kargs):
        mat = self.kernel(x, x)
        self.x = x
        #self.reg.set_params(kernel_params=mat)
        return self.reg.fit(mat, y, *args, **kargs)

    def predict(self, x, *args, **kargs):
        mat = self.kernel(x, self.x)
        #self.reg.set_params(kernel_params=mat)
        return self.reg.predict(mat, *args, **kargs)



class Empty_Regression:
    """
    Class for defining regressions in case they can't be imported
    """
    def __init__(self, *args, **kargs):
        pass



from kernels.pykernels.basic import *
from kernels.pykernels.regular import *



try:
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import LassoLars
    from sklearn.linear_model import LassoLarsCV
    from sklearn.linear_model import LassoLarsIC
    from sklearn.linear_model import MultiTaskLasso
    from sklearn.linear_model import MultiTaskLassoCV
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import ElasticNetCV
    from sklearn.linear_model import MultiTaskElasticNet
    from sklearn.linear_model import MultiTaskElasticNetCV
    from sklearn.linear_model import Lars
    from sklearn.linear_model import LarsCV
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.linear_model import OrthogonalMatchingPursuitCV
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import ARDRegression
    from sklearn.linear_model import SGDRegressor
    from sklearn.linear_model import PassiveAggressiveRegressor
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import TheilSenRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import HuberRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.svm import SVR
except ImportError:
    print("Some regressions packages could not be loaded, running failsafe mode")
    
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError as e:
        print(str(e))
        LinearRegression = Empty_Regression
        
    try:
        from sklearn.linear_model import Ridge
    except ImportError as e:
        print(str(e))
        Ridge = Empty_Regression
        
    try:
        from sklearn.linear_model import RidgeCV
    except ImportError as e:
        print(str(e))
        RidgeCV = Empty_Regression

    try:
        from sklearn.linear_model import Lasso
    except ImportError as e:
        print(str(e))
        Lasso = Empty_Regression
        
    try:
        from sklearn.linear_model import LassoCV
    except ImportError as e:
        print(str(e))
        LassoCV = Empty_Regression
        
    try:
        from sklearn.linear_model import LassoLars
    except ImportError as e:
        print(str(e))
        LassoLars = Empty_Regression
        
    try:
        from sklearn.linear_model import LassoLarsCV
    except ImportError as e:
        print(str(e))
        LassoLarsCV = Empty_Regression
        
    try:
        from sklearn.linear_model import LassoLarsIC
    except ImportError as e:
        print(str(e))
        LassoLarsIC = Empty_Regression
        
    try:
        from sklearn.linear_model import MultiTaskLasso
    except ImportError as e:
        print(str(e))
        MultiTaskLasso = Empty_Regression
        
    try:
        from sklearn.linear_model import MultiTaskLassoCV
    except ImportError as e:
        print(str(e))
        MultiTaskLassoCV = Empty_Regression
        
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError as e:
        print(str(e))
        ElasticNet = Empty_Regression
        
    try:
        from sklearn.linear_model import ElasticNetCV
    except ImportError as e:
        print(str(e))
        ElasticNetCV = Empty_Regression
        
    try:
        from sklearn.linear_model import MultiTaskElasticNet
    except ImportError as e:
        print(str(e))
        MultiTaskElasticNet = Empty_Regression
        
    try:
        from sklearn.linear_model import MultiTaskElasticNetCV
    except ImportError as e:
        print(str(e))
        MultiTaskElasticNetCV = Empty_Regression
        
    try:
        from sklearn.linear_model import Lars
    except ImportError as e:
        print(str(e))
        Lars = Empty_Regression
        
    try:
        from sklearn.linear_model import LarsCV
    except ImportError as e:
        print(str(e))
        LarsCV = Empty_Regression
        
    try:
        from sklearn.linear_model import OrthogonalMatchingPursuit
    except ImportError as e:
        print(str(e))
        OrthogonalMatchingPursuit = Empty_Regression
        
    try:
        from sklearn.linear_model import OrthogonalMatchingPursuitCV
    except ImportError as e:
        print(str(e))
        OrthogonalMatchingPursuitCV = Empty_Regression
        
    try:
        from sklearn.linear_model import BayesianRidge
    except ImportError as e:
        print(str(e))
        BayesianRidge = Empty_Regression
        
    try:
        from sklearn.linear_model import ARDRegression
    except ImportError as e:
        print(str(e))
        ARDRegression = Empty_Regression
        
    try:
        from sklearn.linear_model import SGDRegressor
    except ImportError as e:
        print(str(e))
        SGDRegressor = Empty_Regression
        
    try:
        from sklearn.linear_model import PassiveAggressiveRegressor
    except ImportError as e:
        print(str(e))
        PassiveAggressiveRegressor = Empty_Regression
        
    try:
        from sklearn.linear_model import RANSACRegressor
    except ImportError as e:
        print(str(e))
        RANSACRegressor = Empty_Regression
        
    try:
        from sklearn.linear_model import TheilSenRegressor
    except ImportError as e:
        print(str(e))
        TheilSenRegressor = Empty_Regression
        
    try:
        from sklearn.linear_model import HuberRegressor
    except ImportError as e:
        print(str(e))
        HuberRegressor = Empty_Regression
        
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as e:
        print(str(e))
        RandomForestRegressor = Empty_Regression
        
    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError as e:
        print(str(e))
        ExtraTreesRegressor = Empty_Regression
        
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as e:
        print(str(e))
        GradientBoostingRegressor = Empty_Regression
        
    try:
        from sklearn.kernel_ridge import KernelRidge
    except ImportError as e:
        print(str(e))
        KernelRidge = Empty_Regression
        
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
    except ImportError as e:
        print(str(e))
        GaussianProcessRegressor = Empty_Regression
        
    try:
        from sklearn.svm import SVR
    except ImportError as e:
        print(str(e))
        SVR = Empty_Regression
        


# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
