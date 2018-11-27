import numpy as np
from collections import Counter

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

from utils.base import *



class Split_Dataset(Base_Input_Block):
    """
    Split a dataset into 2 sets, train and test sets
    You need to reset seed to compute a new split (reset it to None if you want)
    Two different splits are unrelated
    If you want each point to appear once and only once into test set, use (Split/Join)_Dataset_N_Parts
    Inputs :
        input_block : Base_Block = NoBlock : input block
        test_size : int - float = 0.1 : test size, in number of examples (int) or proportion (float)
        seed : int = None : random state initialization for ths split
        **kargs : **{} : Base_Input_Block parameters can be set here
    Outputs : 
        (x,y) -> [(x1,y1), (x2,y2)]
    Examples :
        Split_Dataset(in_block)
        Split_Dataset((x,y))
        Split_Dataset((x,y), test_size=10)
        Split_Dataset((x,y), test_size=0.2)
        Split_Dataset((x,y), test_size=0.2, seed=42)
    """
    def __init__(self, dataset=NoBlock, test_size=0.1, seed=None, **kargs):
        super(Split_Dataset, self).__init__(dataset)
        self.params_names = self.params_names.union({"test_size", "seed"})
        self.test_size = test_size
        self.seed = seed
        self.set_params(test_size=test_size, seed=seed, **kargs)

    def set_params(self, *args, **kargs):
        super(Split_Dataset, self).set_params(*args)
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "test_size"):
            changed = (self.test_size != v)
            self.test_size = v
            return changed
        elif (k == "seed") or (k == "random") or (k == "random_state"):
            changed = (self.seed != v) or (self.seed is None)
            self.seed = v
            return changed
        else:
            if (k == "dataset"):
                k = "input"
            return super(Split_Dataset, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "test_size"):
            return self.test_size
        elif (k == "seed") or (k == "random") or (k == "random_state"):
            return self.seed
        else:
            return super(Split_Dataset, self)._get_param(k)

    def split_indexes(self, length):
        if self.seed is not None:
            np.random.seed(self.seed)
        if isinstance(self.test_size, float):
            self.test_size = int(round(length * self.test_size))
        indexes = np.arange(length)
        np.random.shuffle(indexes)
        self.indexes = [indexes[self.test_size:], indexes[:self.test_size]]
        return self.indexes

    def split(self):
        x, y = self._input_block()
        i1, i2 = self.split_indexes(x.shape[0])
        self.output_train = (x[i1], y[i1])
        self.output_test = (x[i2], y[i2])
        self.output = [self.output_train, self.output_test]
        self._update_changed()
        return self.output

    def changed_train(self):
        return self.changed_call()

    def changed_test(self):
        return self.changed_call()

    def changed(self):
        return self.changed_call()

    def _train(self):
        self.split()
        return self.output_train

    def _test(self):
        self.split()
        return self.output_test

    def _call(self):
        self.split()
        return self.output
    


class Sklearn_Split_Dataset(Split_Dataset):
    """
    Same as Split_Dataset, but uses sklearn.cross_validation.train_test_split function to split data
    It also does not provide self.indexes
    """
    def split_indexes(self, length):
        self.indexes = None
        return self.indexes

    def split(self):
        x, y = self._input_block()
        x1, x2, y1, y2 = train_test_split(x, y, test_size=self.test_size, random_state=self.seed)
        self.output_train = (x1, y1)
        self.output_test = (x2, y2)
        self.output = [self.output_train, self.output_test]
        self._update_changed()
        return self.output




class Split_Dataset_N_Parts(Base_Input_Block):
    """
    Split a dataset into n sets
    Inputs :
        input_block : Base_Block = NoBlock : input block
        n_parts : int = 2 : number of datasets at the end
        seed : int = None : random state initialization for this split
        strict_equal_parts : bool = False : if True, removes some lines to have n parts with same size
        **kargs : **{} : Base_Input_Block parameters can be set here
    Outputs : 
        (x,y) -> [(x1,y1), (x2,y2), ..., (xn,yn)]
    Examples :
        Split_Dataset_N_Parts(in_block)
        Split_Dataset_N_Parts((x,y))
        Split_Dataset_N_Parts((x,y), n_parts=10)
        Split_Dataset_N_Parts((x,y), n_parts=10, seed=42)
        Split_Dataset_N_Parts((x,y), n_parts=10, seed=42, strict_equal_parts=True)
    """
    def __init__(self, dataset=NoBlock, n_parts=2, seed=None, strict_equal_parts=False, **kargs):
        super(Split_Dataset_N_Parts, self).__init__(dataset)
        self.params_names = self.params_names.union({"n_parts", "seed", "strict_equal_parts"})
        self.n_parts = n_parts
        self.seed = seed
        self.strict_equal_parts = strict_equal_parts
        self.set_params(n_parts=n_parts, seed=seed, strict_equal_parts=strict_equal_parts, **kargs)

    def set_params(self, *args, **kargs):
        super(Split_Dataset_N_Parts, self).set_params(*args)
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "n_parts"):
            changed = (self.n_parts != v)
            self.n_parts = v
            return changed
        elif (k == "seed") or (k == "random") or (k == "random_state"):
            changed = (self.seed != v) or (self.seed is None)
            self.seed = v
            return changed
        elif (k == "strict_equal_parts"):
            changed = (self.strict_equal_parts != v)
            self.strict_equal_parts = v
            return changed
        else:
            if (k == "dataset"):
                k = "input"
            return super(Split_Dataset_N_Parts, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "n_parts"):
            return self.n_parts
        elif (k == "seed") or (k == "random") or (k == "random_state"):
            return self.seed
        elif (k == "strict_equal_parts"):
            return self.strict_equal_parts
        else:
            return super(Split_Dataset_N_Parts, self)._get_param(k)

    def split_indexes(self, length):
        if self.seed is not None:
            np.random.seed(self.seed)
        indexes = list(range(length))
        np.random.shuffle(indexes)
        part_size = length/self.n_parts
        rest = length - part_size*self.n_parts
        if (rest > 0) and not self.strict_equal_parts:
            add = np.concatenate([np.ones(rest), np.zeros(self.n_parts - rest)])
            np.random.shuffle(add)
            add = [0] + np.cumsum(add).astype(int).tolist()
            limits = [i*part_size+a for i,a in zip(range(self.n_parts + 1), add)]
        else:
            limits = [i*part_size for i in range(self.n_parts + 1)]
        self.indexes = [indexes[limits[i]:limits[i+1]] for i in range(self.n_parts)]
        return self.indexes

    def split(self):
        output = self._input_block_()
        x = output[0]
        y = output[1]
        indexes = self.split_indexes(x.shape[0])
        return [(x[i], y[i]) for i in indexes]

    def compute(self):
        return self.split()



class Join_Dataset_N_Parts(Base_Input_Block):
    """
    Join n datasets into 2 datasets (train and test), the test set is only one of the previous parts
    Inputs :
        input_block : Base_Block = NoBlock : input block
        index : int = 0 : defines the join : test set will be input_block()[index], 
                                             train will be the join of all other datasets
        **kargs : **{} : Base_Inputs_Block parameters can be set here
    Outputs : 
        [(x1,y1), (x2,y2), ..., (xn,yn)] -> [(x1+x2+...+xn-xi, y1+y2+...+yn-yi), (xi,yi)] 
    Examples :
        Join_Dataset_N_Parts(in_block)
        Join_Dataset_N_Parts(in_block, index=3)
    """
    def __init__(self, dataset=NoBlock, index=0, **kargs):
        super(Join_Dataset_N_Parts, self).__init__(dataset)
        self.params_names = self.params_names.union({"index"})
        self.index = index
        self.set_params(index=index, **kargs)

    def set_params(self, *args, **kargs):
        super(Join_Dataset_N_Parts, self).set_params(*args)
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "index"):
            changed = (self.index != v)
            self.index = v
            return changed
        else:
            if (k == "dataset"):
                k = "input"
            return super(Join_Dataset_N_Parts, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "index"):
            return self.index
        else:
            return super(Join_Dataset_N_Parts, self)._get_param(k)

    def split_indexes(self, length):
        if self.seed is not None:
            np.random.seed(self.seed)
        indexes = list(range(length))
        np.random.shuffle(indexes)
        part_size = length/n_parts
        rest = length - part_size*n_parts
        if (rest > 0) and self.equal_parts:
            add = np.concatenate([np.ones(rest), np.zeros(self.n_parts - rest)])
            np.random.shuffle(add)
            add = [0] + np.cumsum(add).astype(int).tolist()
            limits = [i*part_size+a for i,a in zip(range(self.n_parts + 1), add)]
        else:
            limits = [i*part_size for i in range(self.n_parts + 1)]
        self.indexes = [indexes[limits[i]:limits[i+1]] for i in range(self.n_parts)]
        return self.indexes

    def join(self):
        output = self._input_block()
        self.output_test = output[self.index]
        output = zip(*[o for i,o in enumerate(output) if (i != self.index)])
        self.output_train = (np.concatenate(output[0]), np.concatenate(output[1]))
        self.output = [self.output_train, self.output_test]
        self._update_changed()
        return self.output

    def changed_train(self):
        return self.changed_call()

    def changed_test(self):
        return self.changed_call()

    def changed(self):
        return self.changed_call()

    def _train(self):
        self.join()
        return self.output_train

    def _test(self):
        self.join()
        return self.output_test

    def _call(self):
        self.join()
        return self.output



class Non_Uniform_Split_Dataset(Base_Inputs_Block):
    """
    Same as Split_Dataset, but all lines with the same feature f will be put into the same set
    Inputs :
        dataset : Base_Block=NoBlock : input block
        feature : Base_Block=NoBlock : also input block,but is only used to define the split
        test_size : int - float = 0.1 : test size, in number of examples (int) or proportion (float)
        seed : int = None : random state initialization for ths split
        **kargs : **{} : Base_Input_Block parameters can be set here
    Outputs : 
        (x,y) -> [(x1,y1), (x2,y2)]
    Examples :
        Split_Dataset(in_block)
        Split_Dataset((x,y))
        Split_Dataset((x,y), test_size=10)
        Split_Dataset((x,y), test_size=0.2)
        Split_Dataset((x,y), test_size=0.2, seed=42)
    """
    def __init__(self, dataset=NoBlock, feature=NoBlock, test_size=0.1, seed=None, **kargs):
        super(Non_Uniform_Split_Dataset, self).__init__([dataset, feature])
        self.params_names = self.params_names.union({"test_size", "seed"})
        self.test_size = test_size
        self.seed = seed
        self.set_params(test_size=test_size, seed=seed, **kargs)

    def set_params(self, *args, **kargs):
        super(Non_Uniform_Split_Dataset, self).set_params(*args)
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "test_size"):
            changed = (self.test_size != v)
            self.test_size = v
            return changed
        elif (k == "seed") or (k == "random") or (k == "random_state"):
            changed = (self.seed != v) or (self.seed is None)
            self.seed = v
            return changed
        else:
            if (k == "dataset"):
                k = "input_0"
            elif (k == "feature"):
                k = "input_1"
            return super(Non_Uniform_Split_Dataset, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "test_size"):
            return self.test_size
        elif (k == "seed") or (k == "random") or (k == "random_state"):
            return self.seed
        else:
            return super(Non_Uniform_Split_Dataset, self)._get_param(k)

    def split_indexes(self):
        feature = self._input_block[1]()
        if self.seed is not None:
            np.random.seed(self.seed)
        values = np.asarray(list(set(feature)))
        np.random.shuffle(values)
        length = values.shape[0]
        if (length <= 1):
            raise ValueError("Degenerate feature selected for non uniform split")
        if isinstance(self.test_size, float):
            self.test_size = int(round(length * self.test_size))
        if (self.test_size == 0):
            self.test_size = 1
        train_values = set(values[self.test_size:])
        self.indexes_test = []
        self.indexes = []
        for i,feat in enumerate(feature):
            if feat in train_values:
                self.indexes.append(i)
            else:
                self.indexes_test.append(i)
        self.indexes_test = np.asarray(self.indexes_test)
        self.indexes = np.asarray(self.indexes)
        np.random.shuffle(self.indexes)
        np.random.shuffle(self.indexes_test)
        return (self.indexes, self.indexes_test)

    def split(self):
        x, y = self._input_block[0]()
        i1, i2 = self.split_indexes()
        self.output_train = (x[i1], y[i1])
        self.output_test = (x[i2], y[i2])
        self.output = [self.output_train, self.output_test]
        self._update_changed()
        return self.output

    def changed_train(self):
        return self.changed_call()

    def changed_test(self):
        return self.changed_call()

    def changed(self):
        return self.changed_call()

    def _train(self):
        self.split()
        return self.output_train

    def _test(self):
        self.split()
        return self.output_test

    def _call(self):
        self.split()
        return self.output



class Non_Uniform_Split_Dataset_N_Parts(Base_Inputs_Block):

    def __init__(self, dataset=NoBlock, feature=NoBlock, n_parts=2, seed=None, **kargs):
        super(Non_Uniform_Split_Dataset_N_Parts, self).__init__([dataset, feature])
        self.params_names = self.params_names.union({"n_parts", "seed"})
        self.n_parts = n_parts
        self.seed = seed
        self.set_params(n_parts=n_parts, seed=seed, **kargs)

    def set_params(self, *args, **kargs):
        super(Non_Uniform_Split_Dataset_N_Parts, self).set_params(*args)
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "n_parts"):
            changed = (self.n_parts != v)
            self.n_parts = v
            return changed
        elif (k == "seed") or (k == "random") or (k == "random_state"):
            changed = (self.seed != v) or (self.seed is None)
            self.seed = v
            return changed
        else:
            if (k == "dataset"):
                k = "input_0"
            elif (k == "feature"):
                k = "input_1"
            return super(Non_Uniform_Split_Dataset_N_Parts, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "n_parts"):
            return self.n_parts
        elif (k == "seed") or (k == "random") or (k == "random_state"):
            return self.seed
        else:
            return super(Non_Uniform_Split_Dataset_N_Parts, self)._get_param(k)

    def split_indexes(self):
        feature = self._input_block[1]()
        if self.seed is not None:
            np.random.seed(self.seed)
        values = np.asarray(list(set(feature)))
        np.random.shuffle(values)
        length = values.shape[0]
        if (length <= 1):
            raise ValueError("Degenerate feature selected for non uniform split")
        part_size = length/self.n_parts
        rest = length - part_size*self.n_parts
        if (rest > 0):
            add = np.concatenate([np.ones(rest), np.zeros(self.n_parts - rest)])
            np.random.shuffle(add)
            add = [0] + np.cumsum(add).astype(int).tolist()
            limits = [i*part_size+a for i,a in zip(range(self.n_parts + 1), add)]
        else:
            limits = [i*part_size for i in range(self.n_parts + 1)]
        parts_values = [{v:i for v in values[limits[i]:limits[i+1]]} for i in xrange(self.n_parts)]
        dict_values = {}
        for d in parts_values:
            dict_values.update(d)
        self.indexes = [[] for i in xrange(self.n_parts)]
        for i,feat in enumerate(feature):
            self.indexes[dict_values[feat]].append(i)
        self.indexes = [np.asarray(i) for i in self.indexes]
        for i in self.indexes:
            np.random.shuffle(i)
        return self.indexes

    def split(self):
        output = self._input_block_[0]()
        x = output[0]
        y = output[1]
        indexes = self.split_indexes()
        return [(x[i], y[i]) for i in indexes]

    def compute(self):
        return self.split()
