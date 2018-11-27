from utils.base import *



class Custom_Start_Block(Base_Input_Block):

    def __init__(self, func, block_kargs={}, **kargs):
        super(Custom_Start_Block, self).__init__()
        self.func = func
        self.kargs = block_kargs
        self.params_names = self.params_names.union(set(self.kargs.keys()))

    def _set_param(k, v):
        if self.kargs.has_key(k):
            if (self.kargs[k] != v):
                self.kargs[k] = v
                return True
            return False
        else:
            return super(Custom_Start_Block, self)._set_param(k, v)

    def _get_param(k):
        if self.kargs.has_key(k):
            return self.kargs[k]
        else:
            return super(Custom_Start_Block, self)._get_param(k)
        
    def compute(self):
        return self.func(**self.kargs)



class Custom_Input_Block(Base_Input_Block):

    def __init__(self, func, input_block=NoBlock, block_kargs={}, **kargs):
        super(Custom_Input_Block, self).__init__(input_block)
        self.func = func
        self.kargs = block_kargs
        self.params_names = self.params_names.union(set(self.kargs.keys()))

    def _set_param(k, v):
        if self.kargs.has_key(k):
            if (self.kargs[k] != v):
                self.kargs[k] = v
                return True
            return False
        else:
            return super(Custom_Input_Block, self)._set_param(k, v)

    def _get_param(k):
        if self.kargs.has_key(k):
            return self.kargs[k]
        else:
            return super(Custom_Input_Block, self)._get_param(k)
        
    def compute(self):
        return self.func(self._input_block_(), **self.kargs)



class Custom_Inputs_Block(Base_Inputs_Block):

    def __init__(self, func, input_block=[], block_kargs={}, **kargs):
        super(Custom_Inputs_Block, self).__init__(input_block)
        self.func = func
        self.kargs = block_kargs
        self.params_names = self.params_names.union(set(self.kargs.keys()))

    def _set_param(k, v):
        if self.kargs.has_key(k):
            if (self.kargs[k] != v):
                self.kargs[k] = v
                return True
            return False
        else:
            return super(Custom_Inputs_Block, self)._set_param(k, v)
        
    def _get_param(k):
        if self.kargs.has_key(k):
            return self.kargs[k]
        else:
            return super(Custom_Inputs_Block, self)._get_param(k)
        
    def compute(self):
        return self.func(self._input_block_(), **self.kargs)
