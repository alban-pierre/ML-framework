from copy import deepcopy

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

from kernels.pykernels.basic import *
from kernels.pykernels.regular import *

from utils.base import *
from distance.lp_distance import *
from distance.knn_neighbors import *



class Regressor(Base_Input_Block):
    """
    Train and test a regression of scikit learn
    Inputs :
        input_block : Base_Block = [] : dataset to use, it should have different train and test results
        regressor : sk_regressor - inst(sk_regressor) = LinearRegression : sklearn regression to use
        regressor_args : [var] = [] : arguments of the scikit learn regressor to use
        regressor_kargs : {str:var} = {} : keyword arguments of the scikit learn regressor to use
        seed : int = None : random initialization before runnnig the regressor
        **kargs : **{} : Base_Inputs_Block parameters can be set here
    Outputs : 
        train : sk_regressor.predict(dataset.train()[0]) : sk_regressor is fitted on dataset.train()
        test : sk_regressor.predict(dataset.test()[0]) : sk_regressor is fitted on dataset.train()
        call : self.test()
    Examples :
        Regressor(in_block)
        Regressor(in_block, regressor=Ridge)
        Regressor(in_block, regressor=Ridge())
    """
    def __init__(self, dataset=NoBlock, regressor=LinearRegression, regressor_args=[], regressor_kargs={}, seed=None, **kargs):
        super(Regressor, self).__init__(dataset)
        self.params_names = self.params_names.union({"regressor", "seed"})
        if isinstance(regressor, type):
            self.set_params(regressor=regressor, regressor_args=regressor_args, regressor_kargs=regressor_kargs)
        else:
            self.set_params(regressor=regressor)
        self.name += " " + str(self.regressor).split("(")[0]
        self.seed = seed
        self.set_params(seed=seed, **kargs)

    def set_params(self, *args, **kargs):
        super(Regressor, self).set_params(*args)
        changed = False
        if kargs.has_key("regressor"):
            regressor = kargs["regressor"]
            if isinstance(regressor, type):
                regressor_args = []
                if kargs.has_key("regressor_args"):
                    regressor_args = kargs["regressor_args"]
                    del kargs["regressor_args"]
                regressor_kargs = {}
                if kargs.has_key("regressor_kargs"):
                    regressor_kargs = kargs["regressor_kargs"]
                    del kargs["regressor_kargs"]
                self.regressor = regressor(*regressor_args, **regressor_kargs)
            else:
                self.regressor = deepcopy(regressor)
            del kargs["regressor"]
            changed = True
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "regressor"):
            self.regressor = deepcopy(v)
            return True
        elif (k[:11] == "regressor__"):
            self.regressor.set_params(**{k[11:]:v})
            return True
        elif (k[:5] == "reg__"):
            self.regressor.set_params(**{k[5:]:v})
            return True
        elif (k == "seed") or (k == "random") or (k == "random_state"):
            if (self.seed != v) or (self.seed is None):
                self.seed = v
                return True
            return False
        else:
            if (k == "dataset"):
                k = "input"
            return super(Regressor, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "regressor"):
            return self.regressor
        elif (k[:11] == "regressor__"):
            p = self.regressor.get_params()
            if p.has_key(k[11:]):
                return p[k[11:]]
        elif (k[:5] == "reg__"):
            p = self.regressor.get_params()
            if p.has_key(k[5:]):
                return p[k[5:]]
        return super(Regressor, self)._get_param(k)

    def fit(self):
        return self.train()

    def predict(self):
        return self.test()

    def changed_test(self):
        return self.changed_train() or super(Regressor, self).changed_test()
    
    def changed_call(self):
        return self.changed_test()
    
    def _train(self):
        x, y = self._input_block.train()
        if self.seed is not None:
            np.random.seed(self.seed)
        self.regressor.fit(x, y)
        return self.regressor.predict(x)

    def _test(self):
        self.train()
        x, y = self._input_block.test()
        self.output_test = self.regressor.predict(x)
        self.output = [self.output_train, self.output_test]
        self._update_changed_call()
        return self.output_test

    def _call(self):
        self.test()
        return self.output



class Kernel_Regressor(Regressor):

    def __init__(self, dataset=NoBlock, regressor=KernelRidge, kernel=None, regressor_args=[], regressor_kargs={}, **kargs):
        super(Kernel_Regressor, self).__init__(dataset=dataset, regressor=regressor, regressor_args=regressor_args, regressor_kargs=regressor_kargs)
        self.params_names = self.params_names.union({"regressor", "kernel"})
        self.kernel = NoParam
        self.name += " " + str(kernel)
        self.set_params(kernel=kernel, **kargs)

    def _set_param(self, k, v):
        if (k == "kernel"):
            if (v != self.kernel):
                self.kernel = v
                if (self.kernel is None) or isinstance(self.kernel, str):
                    self.regressor.set_params(kernel=self.kernel)
                else:
                    self.regressor.set_params(kernel="precomputed")
                return True
            return False
        elif (k[:8] == "kernel__"):
            if (self.kernel is not None) and not isinstance(self.kernel, str):
                self.kernel = type(self.kernel)(**{k[8:]:v})
                return True
            return False
        else:
            return super(Kernel_Regressor, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "kernel"):
            return self.kernel
        elif (k[:8] == "kernel__"):
            return getattr(self.kernel, k[7:])
        else:
            return super(Kernel_Regressor, self)._get_param(k)

    def fit(self):
        return self.train()

    def predict(self):
        return self.test()

    def _train(self):
        x, y = self._input_block.train()
        self._x = x
        if (self.kernel is not None) and not isinstance(self.kernel, str):
            x = self.kernel(x, x)
        self.regressor.fit(x, y)
        return self.regressor.predict(x)

    def _test(self):
        self.train()
        x, y = self._input_block.test()
        if (self.kernel is not None) and not isinstance(self.kernel, str):
            x = self.kernel(x, self._x)
        self.output_test = self.regressor.predict(x)
        self.output = [self.output_train, self.output_test]
        self._update_changed_call()
        return self.output_test

    def _call(self):
        self.test()
        return self.output



class KNN_Regressor(Base_Input_Block):
    # TODO Upgrade !! We can remove useless computation and use blocks or maybe not
    def __init__(self, dataset=NoBlock, distance='l2', weights=None, k=None, **kargs):
        super(KNN_Regressor, self).__init__(dataset)
        self.params_names = self.params_names.union({"distance", "weights", "k"})
        self.distance = NoParam
        self.weights = weights
        self.knn_indexes = KNN_Indexes(k=k)
        self.set_params(distance=distance, weights=weights, **kargs)

    def _set_param(self, k, v):
        if (k == "distance"):
            if (v != self.distance):
                if (v == 'l2'):
                    self.distance_train = Symetric_Distance_L2()
                    self.distance_test = Asymetric_Distance_L2()
                elif (v == 'precomputed'):
                    self.distance_train = NoBlock
                    self.distance_test = NoBlock
                else:
                    print("Warning : distance {} not available".format(v))
                    self.distance_train = Symetric_Distance_L2()
                    self.distance_test = Asymetric_Distance_L2()
                return True
            return False
        elif (k == "weights") or (k == "coeffs") or (k == "w"):
            if (self.weights != v):
                self.weights = v
                return True
            return False
        elif (k == "k"):
            if (self.knn_indexes.k != v):
                self.knn_indexes.set_params(k=v)
                return True
            return False
        else:
            if (k == "dataset"):
                k = "input"
            return super(KNN_Regressor, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "distance"):
            return self.distance
        elif (k == "weights") or (k == "coeffs") or (k == "w"):
            return self.weights
        elif (k == "k"):
            return self.knn_indexes.k
        else:
            return super(KNN_Regressor, self)._get_param(k)

    def fit(self):
        return self.train()

    def predict(self):
        return self.test()

    def changed_test(self):
        return self.changed_train() or super(KNN_Regressor, self).changed_test()
    
    def changed_call(self):
        return self.changed_test()
    
    def _train(self):
        x, y = self._input_block.train()
        if (self.distance_train is not NoBlock):
            x = self.distance_train(x)
        im = self.knn_indexes(x)
        if (self.weights is None):
            return np.asarray([np.mean(y[i]) for i in im])
        else:
            sh = im.shape[1]
            return np.asarray([np.dot(y[i], self.weights[:sh].T) for i in im])

    def _test(self):
        self.train()
        x, y = self._input_block.train()
        x_, _ = self._input_block.test()
        if (self.distance_train is not NoBlock):
            x_ = self.distance_test(x_, x)
        im = self.knn_indexes(x_)
        if (self.weights is None):
            output = np.asarray([np.mean(y[i]) for i in im])
        else:
            sh = im.shape[1]
            output = np.asarray([np.dot(y[i], self.weights[:sh].T) for i in im])
        self.output_test = output
        self.output = [self.output_train, self.output_test]
        self._update_changed_call()
        return self.output_test

    def _call(self):
        self.test()
        return self.output



class Gaussian_Average_Estimator(Base_Input_Block):

    def __init__(self, input_block=NoBlock, std=1, **kargs):
        super(Gaussian_Average_Estimator, self).__init__(input_block)
        self.params_names = self.params_names.union({"std"})
        self.std = 1.
        self.set_params(std=std, **kargs)

    def _set_param(self, k, v):
        if (k == "std"):
            if (self.std != v):
                self.std = v
                return True
            return False
        else:
            if (k == "dataset"):
                k = "input"
            return super(Gaussian_Average_Estimator, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "std"):
            return self.std
        else:
            return super(Gaussian_Average_Estimator, self)._get_param(k)

    def _train(self):
        x, y = self._input_block.train()
        # self.distance = Symetric_Distance_L2(x)()
        self.distance = symmetric_distance_l2(x, squared=True)
        self.weights = np.exp(-self.distance/float(2*self.std**2))
        self.sum_weights = np.sum(self.weights, axis=0)
        self.coeffs = self.weights/self.sum_weights
        result = np.dot(self.coeffs.transpose(), y)
        return result
        
    def _test(self):
        x1, y1 = self._input_block.train()
        x2, y2 = self._input_block.test()
        # self.distance = l2_dist(x1,x2)#Asymetric_Distance_L2(x1, x2)()
        self.distance = asymmetric_distance_l2(x1, x2, squared=True)
        self.weights = np.exp(-self.distance/float(2*self.std**2))
        self.sum_weights = np.sum(self.weights, axis=0)
        self.coeffs = self.weights/self.sum_weights
        result = np.dot(self.coeffs.transpose(), y1)
        return result
        
    def _call(self):
        return self.test()



def gaussian_average_knn_estimator(data_train, data_test=None, k=10, slf=Empty_Class()):
    x1, y1 = data_train
    if data_test is None:
        x2, y2 = data_train
    else:
        x2, y2 = data_test
    slf.distance = asymmetric_distance_l2(x2, x1)
    slf.sorted_distance = np.sort(slf.distance)
    slf.std_sq = np.mean(slf.sorted_distance[:,:k], axis=1)
    slf.weights = np.exp(-slf.distance/(2.*slf.std_sq)[:,np.newaxis])
    slf.sum_weights = np.sum(slf.weights, axis=1)
    slf.coeffs = slf.weights/slf.sum_weights[:,np.newaxis]
    result = np.dot(slf.coeffs, y1)
    return result


    
class Gaussian_Average_KNN_Estimator(Base_Input_Block):

    def __init__(self, input_block=NoBlock, k=10, **kargs):
        super(Gaussian_Average_KNN_Estimator, self).__init__(input_block)
        self.params_names = self.params_names.union({"k"})
        self.k = 10
        self.set_params(k=k, **kargs)

    def _set_param(self, k, v):
        if (k == "k"):
            if (self.k != v):
                self.k = v
                return True
            return False
        else:
            if (k == "dataset"):
                k = "input"
            return super(Gaussian_Average_KNN_Estimator, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "std"):
            return self.std
        else:
            return super(Gaussian_Average_KNN_Estimator, self)._get_param(k)

    def _train(self):
        x, y = self._input_block.train()
        # self.distance = Symetric_Distance_L2(x)()
        self.distance = symmetric_distance_l2(x)
        # self.distance = np.concatenate([self.distance, 2*self.distance])
        self.sorted_distance = np.sort(self.distance)
        self.std_sq = np.mean(self.sorted_distance[:,:self.k], axis=1)
        self.weights = np.exp(-self.distance/(2.*self.std_sq)[:,np.newaxis])
        self.sum_weights = np.sum(self.weights, axis=1)
        self.coeffs = self.weights/self.sum_weights[:,np.newaxis]
        result = np.dot(self.coeffs, y)
        return result
        
    def _test(self):
        x1, y1 = self._input_block.train()
        x2, y2 = self._input_block.test()
        # self.distance = l2_dist(x2,x1)#Asymetric_Distance_L2(x1, x2)()
        # self.distance = np.concatenate([self.distance, 2*self.distance])
        self.distance = asymmetric_distance_l2(x2, x1)
        self.sorted_distance = np.sort(self.distance)
        self.std_sq = np.mean(self.sorted_distance[:,:self.k], axis=1)
        self.weights = np.exp(-self.distance/(2.*self.std_sq)[:,np.newaxis])
        self.sum_weights = np.sum(self.weights, axis=1)
        self.coeffs = self.weights/self.sum_weights[:,np.newaxis]
        result = np.dot(self.coeffs, y1)
        return result
        
    def _call(self):
        return self.test()
