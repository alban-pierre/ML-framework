import numpy as np

from utils.base import *



def knn_indexes(m, k=None):
    if k is None:
        return np.argsort(m)
    else:
        return np.argsort(m)[:,:k]



class KNN_Indexes(Base_Input_Block):
    
    def __init__(self, input_block=NoBlock, k=None, **kargs):
        super(KNN_Indexes, self).__init__(input_block)
        self.params_names = self.params_names.union({"k"})
        self.k = k
        self.set_params(k=k, **kargs)

    def _set_param(self, k, v):
        if (k == "k"):
            if (self.k != v):
                self.k = v
                return True
            return False
        else:
            return super(KNN_Indexes, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "k"):
            return self.k
        else:
            return super(KNN_Indexes, self)._get_param(k)
        
    def compute(self):
        # return knn_indexes(self._input_block_(), self.k)
        if self.k is None:
            return np.argsort(self._input_block_())
        else:
            return np.argsort(self._input_block_())[:,:self.k]



def knn_distances(m, k=None):
    if k is None:
        return np.sort(m)
    else:
        return np.sort(m)[:,:k]



class KNN_Distances(Base_Input_Block):
    
    def __init__(self, input_block=NoBlock, k=None, **kargs):
        super(KNN_Indexes, self).__init__(input_block)
        self.params_names = self.params_names.union({"k"})
        self.k = k
        self.set_params(k=k, **kargs)

    def _set_param(self, k, v):
        if (k == "k"):
            if (self.k != v):
                self.k = v
                return True
            return False
        else:
            return super(KNN_Distances, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "k"):
            return self.k
        else:
            return super(KNN_Distances, self)._get_param(k)
        
    def compute(self):
        # return knn_distances(self._input_block_(), self.k)
        if self.k is None:
            return np.sort(self._input_block_())
        else:
            return np.sort(self._input_block_())[:,:self.k]
