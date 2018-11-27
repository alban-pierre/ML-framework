from sklearn import preprocessing

from utils.base import *


# Should be obsolete
class Matrix_Dataset(Base_Start_Block):
    """
    DEPRECATED : Matrix_Dataset(x, y) is replaced by Train_Test((x, y))
    Returns a dataset from arrays
    Inputs :
        x : arr(N*D) : features
        y : arr(N) - arr(N*T) : targets
    Outputs : 
        (x,y)
    Examples :
        Matrix_Dataset(x, y)
    """
    def __init__(self, x, y):
        super(Matrix_Dataset, self).__init__()
        self.params_names = self.params_names.union({"x", "y", "dataset"})
        self.dataset = (x, y)
        self.output = self.dataset

    def set_params(self, *args, **kargs):
        super(Matrix_Dataset, self).set_params(*args)
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "x"):
            self.dataset = (v, self.dataset[1])
            return True
        elif (k == "y"):
            self.dataset = (self.dataset[0], v)
            return True
        elif (k == "data") or (k == "dataset") or (k == "input"):
            self.dataset = v
            return True
        else:
            return super(Matrix_Dataset, self)._set_param(k,v)

    def _get_param(self, k):
        if (k == "x"):
            return self.dataset[0]
        elif (k == "y"):
            return self.dataset[1]
        elif (k == "data") or (k == "dataset") or (k == "input"):
            return self.dataset
        else:
            return super(Matrix_Dataset, self)._get_param(k)
        
    def get_data(self):
        return self.dataset

    def compute(self):
        self.output = self.get_data()
        self.output_train = self.output
        self.output_test = self.output
        self._update_changed()
        return self.output



class Sklearn_Dataset(Base_Start_Block):
    """
    Returns a dataset from scikit learn package
    Inputs :
        dataset_name : str = "boston" : dataset name to load
    Outputs : 
        the dataset
    Examples :
        Sklearn_Dataset("boston")
    """
    def __init__(self, dataset_name):
        super(Sklearn_Dataset, self).__init__()
        if (dataset_name == "boston"):
            from sklearn.datasets import load_boston
            # boston = load_boston()
            # self.dataset = (preprocessing.scale(boston.data), preprocessing.scale(boston.target))
            # self.output = self.dataset
            self.all_dataset = load_boston()
            self.dataset = (self.all_dataset["data"], self.all_dataset["target"])
            # self.dataset = (preprocessing.scale(boston.data), preprocessing.scale(boston.target))
            # self.output = self.dataset
            # self._update_changed()
        else:            
            raise ValueError("No such dataset or not implemented")

    def get_data(self):
        return self.dataset

    def compute(self):
        self.output = self.get_data()
        self.output_train = self.output
        self.output_test = self.output
        self._update_changed()
        return self.output
