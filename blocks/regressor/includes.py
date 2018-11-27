
# class Empty_Regressor:
#     """
#     Class for defining regressions in case they can't be imported
#     """
#     def __init__(self, *args, **kargs):
#         pass
Empty_Regressor = None



from kernels.pykernels.basic import *
from kernels.pykernels.regular import *

all_kernels = ['linear', 'poly', 'rbf']
#all_kernels += [RBF, Linear, Polynomial, Cossim, Exponential, Laplacian, RationalQuadratic]
# Polynomial has a problem with Timeout
all_kernels += [RBF, Linear, Cossim, Exponential, Laplacian, RationalQuadratic]
all_kernels += [InverseMultiquadratic, Cauchy, TStudent, ANOVA, Wavelet, Fourier, Tanimoto, Sorensen]
all_kernels += [AdditiveChi2, Chi2, Min, GeneralizedHistogramIntersection, MinMax, Spline, Log, Power]

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
    all_regressors = [LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLars, LassoLarsCV]
    all_regressors += [LassoLarsIC, MultiTaskLasso, MultiTaskLassoCV, ElasticNet, ElasticNetCV]
    all_regressors += [MultiTaskElasticNet, MultiTaskElasticNetCV, Lars, LarsCV]
    all_regressors += [OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, BayesianRidge]
    all_regressors += [ARDRegression, SGDRegressor, PassiveAggressiveRegressor, RANSACRegressor]
    all_regressors += [TheilSenRegressor, RandomForestRegressor, ExtraTreesRegressor]
    all_regressors += [GradientBoostingRegressor, KernelRidge, HuberRegressor]
    all_regressors += [GaussianProcessRegressor, SVR]
except ImportError:
    print("Some regressions packages could not be loaded, running failsafe mode")
    all_regressors = []
    try:
        from sklearn.linear_model import LinearRegression
        all_regressors.append(LinearRegression)
    except ImportError as e:
        print(str(e))
        LinearRegression = Empty_Regressor

    try:
        from sklearn.linear_model import Ridge
        all_regressors.append(Ridge)
    except ImportError as e:
        print(str(e))
        Ridge = Empty_Regressor

    try:
        from sklearn.linear_model import RidgeCV
        all_regressors.append(RidgeCV)
    except ImportError as e:
        print(str(e))
        RidgeCV = Empty_Regressor

    try:
        from sklearn.linear_model import Lasso
        all_regressors.append(Lasso)
    except ImportError as e:
        print(str(e))
        Lasso = Empty_Regressor

    try:
        from sklearn.linear_model import LassoCV
        all_regressors.append(LassoCV)
    except ImportError as e:
        print(str(e))
        LassoCV = Empty_Regressor

    try:
        from sklearn.linear_model import LassoLars
        all_regressors.append(LassoLars)
    except ImportError as e:
        print(str(e))
        LassoLars = Empty_Regressor

    try:
        from sklearn.linear_model import LassoLarsCV
        all_regressors.append(LassoLarsCV)
    except ImportError as e:
        print(str(e))
        LassoLarsCV = Empty_Regressor

    try:
        from sklearn.linear_model import LassoLarsIC
        all_regressors.append(LassoLarsIC)
    except ImportError as e:
        print(str(e))
        LassoLarsIC = Empty_Regressor

    try:
        from sklearn.linear_model import MultiTaskLasso
        all_regressors.append(MultiTaskLasso)
    except ImportError as e:
        print(str(e))
        MultiTaskLasso = Empty_Regressor

    try:
        from sklearn.linear_model import MultiTaskLassoCV
        all_regressors.append(MultiTaskLassoCV)
    except ImportError as e:
        print(str(e))
        MultiTaskLassoCV = Empty_Regressor

    try:
        from sklearn.linear_model import ElasticNet
        all_regressors.append(ElasticNet)
    except ImportError as e:
        print(str(e))
        ElasticNet = Empty_Regressor

    try:
        from sklearn.linear_model import ElasticNetCV
        all_regressors.append(ElasticNetCV)
    except ImportError as e:
        print(str(e))
        ElasticNetCV = Empty_Regressor

    try:
        from sklearn.linear_model import MultiTaskElasticNet
        all_regressors.append(MultiTaskElasticNet)
    except ImportError as e:
        print(str(e))
        MultiTaskElasticNet = Empty_Regressor

    try:
        from sklearn.linear_model import MultiTaskElasticNetCV
        all_regressors.append(MultiTaskElasticNetCV)
    except ImportError as e:
        print(str(e))
        MultiTaskElasticNetCV = Empty_Regressor

    try:
        from sklearn.linear_model import Lars
        all_regressors.append(Lars)
    except ImportError as e:
        print(str(e))
        Lars = Empty_Regressor

    try:
        from sklearn.linear_model import LarsCV
        all_regressors.append(LarsCV)
    except ImportError as e:
        print(str(e))
        LarsCV = Empty_Regressor

    try:
        from sklearn.linear_model import OrthogonalMatchingPursuit
        all_regressors.append(OrthogonalMatchingPursuit)
    except ImportError as e:
        print(str(e))
        OrthogonalMatchingPursuit = Empty_Regressor

    try:
        from sklearn.linear_model import OrthogonalMatchingPursuitCV
        all_regressors.append(OrthogonalMatchingPursuitCV)
    except ImportError as e:
        print(str(e))
        OrthogonalMatchingPursuitCV = Empty_Regressor

    try:
        from sklearn.linear_model import BayesianRidge
        all_regressors.append(BayesianRidge)
    except ImportError as e:
        print(str(e))
        BayesianRidge = Empty_Regressor

    try:
        from sklearn.linear_model import ARDRegression
        all_regressors.append(ARDRegression)
    except ImportError as e:
        print(str(e))
        ARDRegression = Empty_Regressor

    try:
        from sklearn.linear_model import SGDRegressor
        all_regressors.append(SGDRegressor)
    except ImportError as e:
        print(str(e))
        SGDRegressor = Empty_Regressor

    try:
        from sklearn.linear_model import PassiveAggressiveRegressor
        all_regressors.append(PassiveAggressiveRegressor)
    except ImportError as e:
        print(str(e))
        PassiveAggressiveRegressor = Empty_Regressor

    try:
        from sklearn.linear_model import RANSACRegressor
        all_regressors.append(RANSACRegressor)
    except ImportError as e:
        print(str(e))
        RANSACRegressor = Empty_Regressor

    try:
        from sklearn.linear_model import TheilSenRegressor
        all_regressors.append(TheilSenRegressor)
    except ImportError as e:
        print(str(e))
        TheilSenRegressor = Empty_Regressor

    try:
        from sklearn.linear_model import HuberRegressor
        all_regressors.append(HuberRegressor)
    except ImportError as e:
        print(str(e))
        HuberRegressor = Empty_Regressor

    try:
        from sklearn.ensemble import RandomForestRegressor
        all_regressors.append(RandomForestRegressor)
    except ImportError as e:
        print(str(e))
        RandomForestRegressor = Empty_Regressor

    try:
        from sklearn.ensemble import ExtraTreesRegressor
        all_regressors.append(ExtraTreesRegressor)
    except ImportError as e:
        print(str(e))
        ExtraTreesRegressor = Empty_Regressor

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        all_regressors.append(GradientBoostingRegressor)
    except ImportError as e:
        print(str(e))
        GradientBoostingRegressor = Empty_Regressor

    try:
        from sklearn.kernel_ridge import KernelRidge
        all_regressors.append(KernelRidge)
    except ImportError as e:
        print(str(e))
        KernelRidge = Empty_Regressor

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        all_regressors.append(GaussianProcessRegressor)
    except ImportError as e:
        print(str(e))
        GaussianProcessRegressor = Empty_Regressor

    try:
        from sklearn.svm import SVR
        all_regressors.append(Empty_Regressor)
    except ImportError as e:
        print(str(e))
        SVR = Empty_Regressor



# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
