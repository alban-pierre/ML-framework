import numpy as np
from collections import Counter

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split


class Split_Dataset(object):

    def __init__(self):
        pass

    def split_indexes(self, length, test_size=0.1, seed=None):
        np.random.seed(seed)
        if isinstance(test_size, float):
            test_size = int(round(length * test_size))
        self.indexes = np.arange(length)
        np.random.shuffle(self.indexes)
        self.indexes_test = self.indexes[:test_size]
        self.indexes = self.indexes[test_size:]
        return (self.indexes, self.indexes_test)

    def split(self, x, y, test_size=0.1, seed=None):
        i1, i2 = self.split_indexes(x.shape[0], test_size=test_size, seed=seed)
        return x[i1], x[i2], y[i1], y[i2]

    def __call__(self, *args, **kargs):
        return self.split(*args, **kargs)


    
class Sklearn_Split_Dataset(Split_Dataset):

    def split_indexes(self, length, test_size=0.1, seed=None):
        self.indexes_test = None
        self.indexes = None
        return (self.indexes, self.indexes_test)
    
    def split(self, x, y, test_size=0.1, seed=None):
        return train_test_split(x, y, test_size=test_size, random_state=seed)



class Uniform_Split_Dataset(Split_Dataset):

    def __init__(self, feature=('y', 0)):
        self.feature = feature

    def split_indexes(self, labels, test_size=0.1, seed=None):
        np.random.seed(seed)
        count = dict(Counter(labels))
        length = len(labels)
        if isinstance(test_size, float):
            test_size = int(round(length * test_size))
        s = test_size
        ts = length
        ls = {}
        for k,v in count.items():
            fv = v*(float(s)/ts)
            ls[k] = int(fv) + int((np.random.rand(1)[0] < fv - int(fv)))
            ts -= v
            s -= ls[k]
        self.indexes_test = []
        self.indexes = []
        # Two different algorithms, don't know which one is the fastest
        if True: # algo 1
            for k,v in ls.items():
                indexes = np.asarray([i for i,j in enumerate(labels) if j == k])
                np.random.shuffle(indexes)
                self.indexes_test.append(indexes[:v])
                self.indexes.append(indexes[v:])
            self.indexes_test = np.concatenate(self.indexes_test)
            self.indexes = np.concatenate(self.indexes)
        else: # algo 2 does not work yet
            labels = np.copy(labels)
            np.random.shuffle(labels) # inverse shuffle the indexes afterwards !!
            count = {i:0 for i in count.keys()}
            for i,lab in enumerate(labels):
                if (count[lab] < ls[lab]):
                    self.indexes_test.append(i)
                else:
                    self.indexes.append(i)
                count[lab] += 1
            self.indexes_test = np.asarray(self.indexes_test)
            self.indexes = np.asarray(self.indexes)
        # End of the 2 algos
        return (self.indexes, self.indexes_test)

    def split(self, x, y, test_size=0.1, seed=None):
        if isinstance(self.feature, tuple):
            if (self.feature[0] == 'y'):
                if (len(y.shape) == 1):
                    labels = y
                else:
                    labels = y[:,self.feature[1]]
            else:
                labels = x[:,self.feature[1]]
        else:
            labels = self.feature
        i1, i2 = self.split_indexes(labels, test_size=test_size, seed=seed)
        return x[i1], x[i2], y[i1], y[i2]



class Non_Uniform_Split_Dataset(Split_Dataset):

    def __init__(self, feature=('y', 0)):
        self.feature = feature

    def split_indexes(self, labels, test_size=0.1, seed=None):
        np.random.seed(seed)
        values = np.asarray(list(set(labels)))
        np.random.shuffle(values)
        length = len(labels)
        if isinstance(test_size, float):
            test_size = int(round(length * test_size))
        values = set(values[test_size:])
        self.indexes_test = []
        self.indexes = []
        for i,lab in enumerate(labels):
            if lab in values:
                self.indexes.append(i)
            else:
                self.indexes_test.append(i)
        self.indexes_test = np.asarray(self.indexes_test)
        self.indexes = np.asarray(self.indexes)
        return (self.indexes, self.indexes_test)
        
    def split(self, x, y, test_size=0.1, seed=None):
        if isinstance(self.feature, tuple):
            if (self.feature[0] == 'y'):
                if (len(y.shape) == 1):
                    labels = y
                else:
                    labels = y[:,self.feature[1]]
            else:
                labels = x[:,self.feature[1]]
        else:
            labels = self.feature
        i1, i2 = self.split_indexes(labels, test_size=test_size, seed=seed)
        return x[i1], x[i2], y[i1], y[i2]
