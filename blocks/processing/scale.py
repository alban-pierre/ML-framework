import numpy as np
from copy import deepcopy

from sklearn import preprocessing

from utils.base import *



class Inverse_Block(Base_Input_Block):
    
    def __init__(self, original_block=NoBlock, input_block=NoBlock, **kargs):
        super(Inverse_Block, self).__init__(input_block)
        self.original_block = original_block
        self.name = "Inverse " + self.name
        self.set_params(**kargs)

    def changed_train(self):
        if self.original_block._changed_here_train:
            return True
        return self.changed_input_train()

    def changed_test(self):
        if self.original_block._changed_here_test:
            return True
        return self.changed_input_test()

    def changed_call(self):
        if self.original_block._changed_here_call:
            return True
        return self.changed_input_call()

    def changed(self):
        if self.original_block._changed_here_call:
            return True
        if self.original_block._changed_here_test or self._changed_here_train:
            return True
        return self.changed_input()
        
    def compute(self):
        return self.original_block.inverse(self._input_block_())



class Scaler_xy(Base_Input_Block):
    # TODO scaler when data is only one matrix, ie when there are no test
    def __init__(self, dataset=NoBlock, **kargs):
        super(Scaler_xy, self).__init__(dataset)
        self.scaler_x = preprocessing.StandardScaler()
        self.scaler_y = preprocessing.StandardScaler()
        self.set_params(**kargs)

    def fit(self):
        return self.train()

    def transform(self):
        return self.test()
    
    def changed_test(self):
        return self.changed_train() or super(Scaler_xy, self).changed_test()
    
    def changed_call(self):
        return self.changed_test()
    
    def _train(self):
        x, y = deepcopy(self._input_block.train())
        y = y[:, np.newaxis]
        self.scaler_x = self.scaler_x.fit(x)
        self.scaler_y = self.scaler_y.fit(y)
        return (self.scaler_x.transform(x), self.scaler_y.transform(y)[:,0])

    def _test(self):
        self.train()
        x, y = deepcopy(self.input_block.test())
        y = y[:, np.newaxis]
        self.output_test = (self.scaler_x.transform(x), self.scaler_y.transform(y)[:,0])
        self.output = [self.output_train, self.output_test]
        self._update_changed_call()
        return self.output_test

    def _call(self):
        self.test()
        return self.output

    def inverse(self, data):
        x, y = data
        return (self.scaler_x.inverse_transform(x), self.scaler_y.inverse_transform(y))



class Scaler_x(Base_Input_Block):

    def __init__(self, dataset=NoBlock, **kargs):
        super(Scaler_x, self).__init__(dataset)
        self.scaler_x = preprocessing.StandardScaler()
        self.set_params(**kargs)

    def fit(self):
        return self.train()

    def transform(self):
        return self.test()

    def changed_test(self):
        return self.changed_train() or super(Scaler_x, self).changed_test()
    
    def changed_call(self):
        return self.changed_test()
    
    def _train(self):
        x, y = deepcopy(self._input_block.train())
        self.scaler_x = self.scaler_x.fit(x)
        return (self.scaler_x.transform(x), y)

    def _test(self):
        self.train()
        x, y = deepcopy(self.input_block.test())
        self.output_test = (self.scaler_x.transform(x), y)
        self.output = [self.output_train, self.output_test]
        self._update_changed_call()
        return self.output_test

    def _call(self):
        self.test()
        return self.output

    def inverse(self, data):
        x, y = data
        return (self.scaler_x.inverse_transform(x), y)
