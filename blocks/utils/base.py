from itertools import count



class Empty_Class(object):
    pass



class _NoParamType(object):
    """
    Class NoParam, it behaves like a second None
    It is used as a default value when None is a possible value of the param
    """
    def __new__(cls):
        return NoParam

    def __reduce__(self):
        return (_NoParamType, ())

    def __copy__(self):
        return NoParam

    def __deepcopy__(self, memo):
        return NoParam

    def __call__(self, default):
        pass



try:
    NoParam
except NameError:
    NoParam = object.__new__(_NoParamType)



# def get_name(inst):
#     """
#     Get shorter name of a class from its __str__ method
#     """
#     s = str(inst)
#     return s.split()[0].split('.')[-1]


class Base_Block(object):
    """
    Base Block, every block inherit from this one
    As there are quite tranparent blocks, this one does not define default methods or attributes
    """
    pass
    # # DEBUG OPTION, IMPORTANT TO KEEP
    # def __getattribute__(self, attr):
    #     if (attr == "train"):
    #         print("{} : train".format(self.name))
    #     elif (attr == "test"):
    #         print("{} : test".format(self.name))
    #     elif (attr == "__call__"):
    #         print("{} : call".format(self.name))
    #     if (attr == "_train"):
    #         print("{} : _train".format(self.name))
    #     elif (attr == "_test"):
    #         print("{} : _test".format(self.name))
    #     elif (attr == "_call"):
    #         print("{} : _call".format(self.name))
    #     elif (attr == "compute"):
    #         print("{} : compute".format(self.name))
    #     return super(Base_Block, self).__getattribute__(attr)
    # # DEBUG OPTION, IMPORTANT TO KEEP



class Base_Id_Block(Base_Block):
    """
    Base Id, Block, every block that transform the input inherit from this class
    It defines _ids, _ids_train and _ids_test, so that the block computation is lazy by default
    It also defines __copy__, __deepcopy__, and instance name
    """
    _ids = count(3)
    _ids_train = count(3)
    _ids_test = count(3)

    def __new__(typ, *args, **kwargs):
        obj = super(Base_Block, typ).__new__(typ, *args, **kwargs)
        obj._id = next(obj._ids)
        obj._id_train = next(obj._ids_train)
        obj._id_test = next(obj._ids_test)
        return obj

    def __init__(self):
        self.name = self.__class__.__name__ # get_name(self)
        self._ct = 2

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        _id = result._id
        _id_train = result._id_train
        _id_test = result._id_test
        result.__dict__.update(self.__dict__)
        result._id = _id
        result._id_train = _id_train
        result._id_test = _id_test
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        _id = result._id
        _id_train = result._id_train
        _id_test = result._id_test
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        result._id = _id
        result._id_train = _id_train
        result._id_test = _id_test
        return result

    def set_params(self, *args, **kargs):
        return False

    def _set_param(self, *args, **kargs):
        return False

    def get_params(self, *args, **kargs):
        return False

    def _get_param(self, *args, **kargs):
        return False

    def _init_changed(self):
        pass

    def _update_changed_train(self):
        pass

    def _update_changed_test(self):
        pass

    def _update_changed_call(self):
        pass

    def _update_changed(self):
        pass

    def changed_train(self):
        return self.changed()

    def changed_test(self):
        return self.changed()

    def changed_call(self):
        return self.changed()

    def changed(self):
        return False

    # DEBUG OPTION, IMPORTANT TO KEEP
    # def __getattribute__(self, attr):
    #     if (attr == "train"):
    #         print("{} : train".format(self.name))
    #     elif (attr == "test"):
    #         print("{} : test".format(self.name))
    #     elif (attr == "__call__"):
    #         print("{} : call".format(self.name))
    #     if (attr == "_train"):
    #         print("{} : _train".format(self.name))
    #     elif (attr == "_test"):
    #         print("{} : _test".format(self.name))
    #     elif (attr == "_call"):
    #         print("{} : _call".format(self.name))
    #     elif (attr == "compute"):
    #         print("{} : compute".format(self.name))
    #     return super(Base_Block, self).__getattribute__(attr)
    # DEBUG OPTION, IMPORTANT TO KEEP



class _NoBlockType(Base_Block):
    """
    Default value for input block, meaning "there are no input block defined yet"
    Consequently, when a block has NoBlock as an input, you cannot compute it
    """
    def __new__(cls):
        return NoBlock

    def __reduce__(self):
        return (_NoBlockType, ())

    def __copy__(self):
        return NoBlock

    def __deepcopy__(self, memo):
        return NoBlock

    def train(self, default):
        pass

    def test(self, default):
        pass

    def __call__(self, default):
        pass



try:
    NoBlock
except NameError:
    NoBlock = object.__new__(_NoBlockType)



#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################



class Base_Start_Block(Base_Id_Block):
    """
    Base Block that only computes something from parameters, it has no input blocks
    All blocks inheriting from this one must call this class methods using "super" for methods :
        __init__, _set_param, _get_param
    A block inheriting from this class can define :
        "compute" method if the output is the same when we call "train()", "test()" or "()"
        "_train", "_test" and "_call" methods when we want to define different outputs
            Note that the update of _ids is done automatically when calling "train", "test" or "()"
        "__init__", "_set_param", "_get_param" when you want your block to have some intern parameters
        "changed_train", "changed_test", "changed_call" when you want to change default lazyness
    """    
    def __init__(self, **kargs):
        super(Base_Start_Block, self).__init__()
        self.params_names = {"name"}
        # self.all_params = self.params
        self._init_changed()
        if kargs:
            self.set_params(**kargs)

    def set_params(self, *args, **kargs):
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "changed_here_train"):
            self.set_changed_here_train(v)
            return False
        elif (k == "changed_here_test"):
            self.set_changed_here_test(v)
            return False
        elif (k == "changed_here_call"):
            self.set_changed_here_call(v)
            return False
        elif (k == "changed_here"):
            self.set_changed_here(v)
            return False
        elif (k == "name"):
            self.name = v
            return False
        print("Warning : could not set attribute {} for block {}".format(k, self.name))
        return False

    def get_params(self, *args, **kargs):
        if args:
            res = (self._get_param(k) for k in args)
            if kargs.has_key("as_dict") and kargs["as_dict"]:
                return {k:v for k,v in zip(args, res)}
            else:
                res = tuple(res)
                return (res[0] if (len(res) == 1) else res)
        else:
            return {k:self._get_param(k) for k in self.params_names}

    def _get_param(self, k):
        if (k == "name"):
            return self.name
        return NoParam

    def __getattr__(self, k):
        if (k[:4] == "set_"):
            return (lambda v : self.set_params(**{k[4:]:v}))
        elif (k[:4] == "get_"):
            return (lambda : self.get_params(k[4:]))
        else:
            # return getattr(super(Base_Start_Block, self), k)
            cls_name = self.__class__.__name__
            raise AttributeError("'{}' object has no attribute '{}'".format(cls_name, k))
        
    def _init_changed(self):
        self._changed_here_train = True
        self._changed_here_test = True
        self._changed_here_call = True

    def _update_changed_train(self):
        self._changed_here_train = False
        self._id_train = next(self._ids_train)

    def _update_changed_test(self):
        self._changed_here_test = False
        self._id_test = next(self._ids_test)

    def _update_changed_call(self):
        self._changed_here_call = False
        self._id = next(self._ids)

    def _update_changed(self):
        self._update_changed_train()
        self._update_changed_test()
        self._update_changed_call()

    def changed_train(self):
        return self._changed_here_train

    def changed_test(self):
        return self._changed_here_test

    def changed_call(self):
        return self._changed_here_call

    def changed(self):
        return self._changed_here_call or self._changed_here_test or self._changed_here_train

    def set_changed_here_train(self, changed_here_train):
        self._changed_here_train = self._changed_here_train or changed_here_train

    def set_changed_here_test(self, changed_here_test):
        self._changed_here_test = self._changed_here_test or changed_here_test

    def set_changed_here_call(self, changed_here_call):
        self._changed_here_call = self._changed_here_call or changed_here_call

    def set_changed_here(self, changed_here):
        self._changed_here_train = self._changed_here_train or changed_here
        self._changed_here_test = self._changed_here_test or changed_here
        self._changed_here_call = self._changed_here_call or changed_here
        
    def train(self, *args, **kargs):
        if args or kargs:
            self.set_params(*args, **kargs)
        if self.changed_train():
            self.output_train = self._train()
            self._update_changed_train()
        return self.output_train

    def _train(self):
        return self.compute()
        
    def test(self, *args, **kargs):
        if args or kargs:
            self.set_params(*args, **kargs)
        if self.changed_test():
            self.output_test = self._test()
            self._update_changed_test()
        return self.output_test

    def _test(self):
        return self.compute()
        
    def __call__(self, *args, **kargs):
        if args or kargs:
            self.set_params(*args, **kargs)
        if self.changed_call():
            self.output = self._call()
            self._update_changed_call()
        return self.output

    def _call(self):
        return self.compute()

    def compute(self):
        return None



#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################



class _Input_Block_For_Class_Base_Input_Block(object):
    """
    Class that defines "train", "test" and "()" for any input, Base_Block or array, list etc
    """
    def __init__(self, mother_class):
        self.mother_class = mother_class
    def train(self):
        if isinstance(self.mother_class.input_block, Base_Block):
            return self.mother_class.input_block.train()
        else:
            return self.mother_class.input_block
    def test(self):
        if isinstance(self.mother_class.input_block, Base_Block):
            return self.mother_class.input_block.test()
        else:
            return self.mother_class.input_block
    def __call__(self):
        if isinstance(self.mother_class.input_block, Base_Block):
            return self.mother_class.input_block()
        else:
            return self.mother_class.input_block



class Base_Input_Block(Base_Id_Block):
    """
    Base Block that only computes something from parameters and one input block
    All blocks inheriting from this one must call this class methods using "super" for methods :
        __init__, _set_param, _get_param
    A block inheriting from this class can define :
        "compute" method if the output is the same when we call "train()", "test()" or "()"
        "_train", "_test" and "_call" methods when we want to define different outputs
            Note that the update of _ids is done automatically when calling "train", "test" or "()"
        "__init__", "_set_param", "_get_param" when you want your block to have some intern parameters
        "changed_train", "changed_test", "changed_call" when you want to change default lazyness
    """    
    def __init__(self, input_block=NoBlock, **kargs):
        super(Base_Input_Block, self).__init__()
        self.params_names = {"input_block", "name"}
        self._init_changed()
        self.input_block = input_block
        self._input_block = _Input_Block_For_Class_Base_Input_Block(self)
        if kargs:
            self.set_params(**kargs)

    def set_params(self, *args, **kargs):
        changed = False
        for v in args:
            changed = self._set_param("input", v) or changed
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "changed_here_train"):
            self.set_changed_here_train(v)
            return False
        elif (k == "changed_here_test"):
            self.set_changed_here_test(v)
            return False
        elif (k == "changed_here_call"):
            self.set_changed_here_call(v)
            return False
        elif (k == "changed_here"):
            self.set_changed_here(v)
            return False
        elif (k == "name"):
            self.name = v
            return False
        elif (k[:5] == "input"):
            if (v is not self.input_block):
                self.input_block = v
                self._last_input_id = 0
                self._last_input_id_train = 0
                self._last_input_id_test = 0
            return False
        print("Warning : could not set attribute {} for block {}".format(k, self.name))
        return False

    def get_params(self, *args, **kargs):
        if args:
            res = (self._get_param(k) for k in args)
            if kargs.has_key("as_dict") and kargs["as_dict"]:
                return {k:v for k,v in zip(args, res)}
            else:
                res = tuple(res)
                return (res[0] if (len(res) == 1) else res)
        else:
            return {k:self._get_param(k) for k in self.params_names}

    def _get_param(self, k):
        if (k[:5] == "input"):
            return self.input_block
        elif (k == "name"):
            return self.name
        return NoParam

    def __getattr__(self, k):
        if (k[:4] == "set_"):
            return (lambda v : self.set_params(**{k[4:]:v}))
        elif (k[:4] == "get_"):
            return (lambda : self.get_params(k[4:]))
        else:
            # return getattr(super(Base_Start_Block, self), k)
            cls_name = self.__class__.__name__
            raise AttributeError("'{}' object has no attribute '{}'".format(cls_name, k))
        
    def _init_changed(self):
        self._changed_here_train = True
        self._changed_here_test = True
        self._changed_here_call = True
        self._last_input_id_train = 0
        self._last_input_id_test = 0
        self._last_input_id = 0

    def _update_changed_train(self):
        self._changed_here_train = False
        self._id_train = next(self._ids_train)
        if hasattr(self.input_block, "_id"):
            self._last_input_id_train = self.input_block._id_train
        else:
            self._last_input_id_train = 2

    def _update_changed_test(self):
        self._changed_here_test = False
        self._id_test = next(self._ids_test)
        if hasattr(self.input_block, "_id"):
            self._last_input_id_test = self.input_block._id_test
        else:
            self._last_input_id_test = 2

    def _update_changed_call(self):
        self._changed_here_call = False
        self._id = next(self._ids)
        if hasattr(self.input_block, "_id"):
            self._last_input_id = self.input_block._id
        else:
            self._last_input_id = 2
        
    def _update_changed(self):
        self._update_changed_train()
        self._update_changed_test()
        self._update_changed_call()

    def changed_train(self):
        return self._changed_here_train or self.changed_input_train()

    def changed_test(self):
        return self._changed_here_test or self.changed_input_test()

    def changed_call(self):
        return self._changed_here_call or self.changed_input_call()

    def changed(self):
        if self._changed_here_call or self._changed_here_test or self._changed_here_train:
            return True
        return self.changed_input()

    def changed_input_train(self):
        if hasattr(self.input_block, "_id"):
            if (self.input_block._id_train != self._last_input_id_train):
                return True
            return self.input_block.changed_train()
        return (self._last_input_id_train == 0)

    def changed_input_test(self):
        if hasattr(self.input_block, "_id"):
            if (self.input_block._id_test != self._last_input_id_test):
                return True
            return self.input_block.changed_test()
        return (self._last_input_id_test == 0)

    def changed_input_call(self):
        if hasattr(self.input_block, "_id"):
            if (self.input_block._id != self._last_input_id):
                return True
            return self.input_block.changed_call()
        return (self._last_input_id == 0)

    def changed_input(self):
        return self.changed_input_call() or self.changed_input_test() or self.changed_input_train()

    def set_changed_here_train(self, changed_here_train):
        self._changed_here_train = self._changed_here_train or changed_here_train

    def set_changed_here_test(self, changed_here_test):
        self._changed_here_test = self._changed_here_test or changed_here_test

    def set_changed_here_call(self, changed_here_call):
        self._changed_here_call = self._changed_here_call or changed_here_call

    def set_changed_here(self, changed_here):
        self._changed_here_train = self._changed_here_train or changed_here
        self._changed_here_test = self._changed_here_test or changed_here
        self._changed_here_call = self._changed_here_call or changed_here

    def _input_block_(self):
        if isinstance(self.input_block, Base_Block):
            if (self._ct == 0):
                return self.input_block.train()
            elif (self._ct == 1):
                return self.input_block.test()
            else:
                return self.input_block()
        else:
            return self.input_block

    def train(self, *args, **kargs):
        self._ct = 0
        if args or kargs:
            self.set_params(*args, **kargs)
        if self.changed_train():
            self.output_train = self._train()
            self._update_changed_train()
        return self.output_train

    def _train(self):
        return self.compute()
        
    def test(self, *args, **kargs):
        self._ct = 1
        if args or kargs:
            self.set_params(*args, **kargs)
        if self.changed_test():
            self.output_test = self._test()
            self._update_changed_test()
        return self.output_test

    def _test(self):
        return self.compute()
        
    def __call__(self, *args, **kargs):
        self._ct = 2
        if args or kargs:
            self.set_params(*args, **kargs)
        if self.changed_call():
            self.output = self._call()
            self._update_changed_call()
        return self.output

    def _call(self):
        return self.compute()

    def compute(self):
        return self._input_block_()



#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################



class _Inputs_Block_For_Class_Base_Inputs_Block_(object):
    """
    Class that defines "()" for any input_list, Base_Block or array, list etc
    """
    def __init__(self, mother_class):
        self.mother_class = mother_class
    def __getitem__(self, i): # TODO allow slices
        return _Input_Block_For_Class_Base_Inputs_Block_(self.mother_class, i)
    def __call__(self):
        return [self.__getitem__(i)() for i in xrange(len(self.mother_class.input_block))]



class _Input_Block_For_Class_Base_Inputs_Block_(object):
    """
    Class that defines "()" for any input_i, Base_Block or array, list etc
    """
    def __init__(self, mother_class, i):
        self.mother_class = mother_class
        self.i = i
    def __call__(self):
        if isinstance(self.mother_class.input_block[self.i], Base_Block):
            if (self.mother_class._ct == 0):
                return self.mother_class.input_block[self.i].train()
            elif (self.mother_class._ct == 1):
                return self.mother_class.input_block[self.i].test()
            else:
                return self.mother_class.input_block[self.i]()
        else:
            return self.mother_class.input_block[self.i]



class _Inputs_Block_For_Class_Base_Inputs_Block(object):
    """
    Class that defines "train", "test" and "()" for any input_list, Base_Block or array, list etc
    """
    def __init__(self, mother_class):
        self.mother_class = mother_class
    def __getitem__(self, i):
        return _Input_Block_For_Class_Base_Inputs_Block(self.mother_class, i)
    def train(self):
        return [self.__getitem__(i).train() for i in xrange(len(self.mother_class.input_block))]
    def test(self):
        return [self.__getitem__(i).test() for i in xrange(len(self.mother_class.input_block))]
    def __call__(self):
        return [self.__getitem__(i)() for i in xrange(len(self.mother_class.input_block))]



class _Input_Block_For_Class_Base_Inputs_Block(object):
    """
    Class that defines "train", "test" and "()" for any input_i, Base_Block or array, list etc
    """
    def __init__(self, mother_class, i):
        self.mother_class = mother_class
        self.i = i
    def train(self):
        if isinstance(self.mother_class.input_block[self.i], Base_Block):
            return self.mother_class.input_block[self.i].train()
        else:
            return self.mother_class.input_block[self.i]
    def test(self):
        if isinstance(self.mother_class.input_block[self.i], Base_Block):
            return self.mother_class.input_block[self.i].test()
        else:
            return self.mother_class.input_block[self.i]
    def __call__(self):
        if isinstance(self.mother_class.input_block[self.i], Base_Block):
            return self.mother_class.input_block[self.i]()
        else:
            return self.mother_class.input_block[self.i]



class Base_Inputs_Block(Base_Id_Block):
    """
    Base Block that only computes something from parameters and a list of input blocks
    All blocks inheriting from this one must call this class methods using "super" for methods :
        __init__, _set_param, _get_param
    A block inheriting from this class can define :
        "compute" method if the output is the same when we call "train()", "test()" or "()"
        "_train", "_test" and "_call" methods when we want to define different outputs
            Note that the update of _ids is done automatically when calling "train", "test" or "()"
        "__init__", "_set_param", "_get_param" when you want your block to have some intern parameters
        "changed_train", "changed_test", "changed_call" when you want to change default lazyness
    """
    def __init__(self, input_block=NoParam, *args, **kargs):
        super(Base_Inputs_Block, self).__init__()
        self.params_names = {"input_block", "name"}
        if (input_block is NoParam):
            self.input_block = []
        elif args:
            self.input_block = [input_block] + list(args)
        elif not isinstance(input_block, list):
            self.input_block = [input_block]
        elif any([isinstance(i, Base_Block) for i in input_block]):
            self.input_block = input_block
        else:
            self.input_block = [input_block]
        self._init_changed()
        self._input_block = _Inputs_Block_For_Class_Base_Inputs_Block(self)
        self._input_block_ = _Inputs_Block_For_Class_Base_Inputs_Block_(self)
        if kargs:
            self.set_params(**kargs)

    def set_params(self, *args, **kargs):
        changed = False
        if args:
            if len(args) > 1:
                input_block = args
            elif not isinstance(args[0], list):
                input_block = args
            elif any([isinstance(i, Base_Block) for i in args[0]]):
                input_block = args[0]
            else:
                input_block = args
            for i,v in enumerate(args):
                changed = self._set_param("input_"+str(i), v) or changed
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "changed_here_train"):
            self.set_changed_here_train(v)
            return False
        elif (k == "changed_here_test"):
            self.set_changed_here_test(v)
            return False
        elif (k == "changed_here_call"):
            self.set_changed_here_call(v)
            return False
        elif (k == "changed_here"):
            self.set_changed_here(v)
            return False
        elif (k == "name"):
            self.name = v
            return False
        elif (k[:5] == "input"):
            if (k == "input"):
                self.input_block = v
                self._last_input_id_train = [0 for b in self.input_block]
                self._last_input_id_test = [0 for b in self.input_block]
                self._last_input_id = [0 for b in self.input_block]
            else:
                i = int(k[6:])
                while (len(self.input_block) <= i):
                    self.input_block.append(NoBlock)
                    self._last_input_id.append(0)
                    self._last_input_id_train.append(0)
                    self._last_input_id_test.append(0)
                if (v is not self.input_block[i]):
                    self.input_block[i] = v
                    self._last_input_id[i] = 0
                    self._last_input_id_train[i] = 0
                    self._last_input_id_test[i] = 0
            return False
        print("Warning : could not set attribute {} for block {}".format(k, self.name))
        return False

    def get_params(self, *args, **kargs):
        if args:
            res = (self._get_param(k) for k in args)
            if kargs.has_key("as_dict") and kargs["as_dict"]:
                return {k:v for k,v in zip(args, res)}
            else:
                res = tuple(res)
                return (res[0] if (len(res) == 1) else res)
        else:
            return {k:self._get_param(k) for k in self.params_names}

    def _get_param(self, k):
        if (k[:5] == "input"):
            if (k == "input") or (k == "input_block"):
                return self.input_block
            else:
                i = int(k[6:])
                return self.input_block[i]
        elif (k == "name"):
            return self.name
        return NoParam

    def __getattr__(self, k):
        if (k[:4] == "set_"):
            return (lambda v : self.set_params(**{k[4:]:v}))
        elif (k[:4] == "get_"):
            return (lambda : self.get_params(k[4:]))
        else:
            # return getattr(super(Base_Start_Block, self), k)
            cls_name = self.__class__.__name__
            raise AttributeError("'{}' object has no attribute '{}'".format(cls_name, k))
        
    def _init_changed(self):
        self._changed_here_train = True
        self._changed_here_test = True
        self._changed_here_call = True
        self._last_input_id_train = [0 for b in self.input_block]
        self._last_input_id_test = [0 for b in self.input_block]
        self._last_input_id = [0 for b in self.input_block]

    def _update_changed_train(self):
        self._changed_here_train = False
        self._id_train = next(self._ids_train)
        self._last_input_id_train = [(b._id_train if hasattr(b, "_id") else 2) for b in self.input_block]

    def _update_changed_test(self):
        self._changed_here_test = False
        self._id_test = next(self._ids_test)
        self._last_input_id_test = [(b._id_test if hasattr(b, "_id") else 2) for b in self.input_block]

    def _update_changed_call(self):
        self._changed_here_call = False
        self._id = next(self._ids)
        self._last_input_id = [(b._id if hasattr(b, "_id") else 2) for b in self.input_block]

    def _update_changed(self):
        self._update_changed_train()
        self._update_changed_test()
        self._update_changed_call()

    def changed_train(self, val=None):
        if self._changed_here_train:
            return True
        return self.changed_input_train(val)

    def changed_test(self, val=None):
        if self._changed_here_test:
            return True
        return self.changed_input_test(val)

    def changed_call(self, val=None):
        if self._changed_here_call:
            return True
        return self.changed_input_call(val)

    def changed(self, val=None):
        if self._changed_here_call or self._changed_here_test or self._changed_here_train:
            return True
        return self.changed_input(val)

    def changed_input_train(self, val=None):
        if (val is None):
            for b,c in zip(self.input_block, self._last_input_id_train):
                if hasattr(b, "_id"):
                    if (b._id_train != c) or b.changed_train():
                        return True
                elif (c == 0):
                    return True
        elif isinstance(val, int):
            if hasattr(self.input_block[val], "_id"):
                if (self.input_block[val]._id_train != self._last_input_id_train[val]):
                    return True
                if self.input_block[val].changed_train():
                    return True
            elif (self._last_input_id_train[val] == 0):
                return True
        else:
            for b,c in zip(self.input_block[val], self._last_input_id_train[val]):
                if hasattr(b, "_id"):
                    if (b._id_train != c) or b.changed_train():
                        return True
                elif (c == 0):
                    return True
        return False

    def changed_input_test(self, val=None):
        if (val is None):
            for b,c in zip(self.input_block, self._last_input_id_test):
                if hasattr(b, "_id"):
                    if (b._id_test != c) or b.changed_test():
                        return True
                elif (c == 0):
                    return True
        elif isinstance(val, int):
            if hasattr(self.input_block[val], "_id"):
                if (self.input_block[val]._id_test != self._last_input_id_test[val]):
                    return True
                if self.input_block[val].changed_test():
                    return True
            elif (self._last_input_id_test[val] == 0):
                return True
        else:
            for b,c in zip(self.input_block[val], self._last_input_id_test[val]):
                if hasattr(b, "_id"):
                    if (b._id_test != c) or b.changed_test():
                        return True
                elif (c == 0):
                    return True
        return False

    def changed_input_call(self, val=None):
        if (val is None):
            for b,c in zip(self.input_block, self._last_input_id):
                if hasattr(b, "_id"):
                    if (b._id != c) or b.changed_call():
                        return True
                elif (c == 0):
                    return True
        elif isinstance(val, int):
            if hasattr(self.input_block[val], "_id"):
                if (self.input_block[val]._id != self._last_input_id[val]):
                    return True
                if self.input_block[val].changed_call():
                    return True
            elif (self._last_input_id[val] == 0):
                return True
        else:
            for b,c in zip(self.input_block[val], self._last_input_id[val]):
                if hasattr(b, "_id"):
                    if (b._id != c) or b.changed_call():
                        return True
                elif (c == 0):
                    return True
        return False

    def changed_input(self, val=None):
        return self.changed_input_call(val) or self.changed_input_test(val) or self.changed_input_train(val)

    def set_changed_here_train(self, changed_here_train):
        self._changed_here_train = self._changed_here_train or changed_here_train

    def set_changed_here_test(self, changed_here_test):
        self._changed_here_test = self._changed_here_test or changed_here_test

    def set_changed_here_call(self, changed_here_call):
        self._changed_here_call = self._changed_here_call or changed_here_call

    def set_changed_here(self, changed_here):
        self._changed_here_train = self._changed_here_train or changed_here
        self._changed_here_test = self._changed_here_test or changed_here
        self._changed_here_call = self._changed_here_call or changed_here

    def train(self, *args, **kargs):
        self._ct = 0
        if args or kargs:
            self.set_params(*args, **kargs)
        if self.changed_train():
            self.output_train = self._train()
            self._update_changed_train()
        return self.output_train

    def _train(self):
        return self.compute()
        
    def test(self, *args, **kargs):
        self._ct = 1
        if args or kargs:
            self.set_params(*args, **kargs)
        if self.changed_test():
            self.output_test = self._test()
            self._update_changed_test()
        return self.output_test

    def _test(self):
        return self.compute()
        
    def __call__(self, *args, **kargs):
        self._ct = 2
        if args or kargs:
            self.set_params(*args, **kargs)
        if self.changed_call():
            self.output = self._call()
            self._update_changed_call()
        return self.output

    def _call(self):
        return self.compute()

    def compute(self):
        return self._input_block_()



#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################



_Input_Block_For_Class_Transparent_Block = _Input_Block_For_Class_Base_Input_Block



class Transparent_Block(Base_Block):
    """
    Base Block where output is equal to input and it does not have "changed" functions
    All blocks inheriting from this one must call this class methods using "super" for methods :
        __init__, _set_param, _get_param
    A block inheriting from this class can define :
        "compute" method if the output is the same when we call "train()", "test()" or "()"
        "_train", "_test" and "_call" methods when we want to define different outputs
        "__init__", "_set_param", "_get_param" when you want your block to have some intern parameters
    """    
    def __init__(self, input_block=NoBlock, **kargs):
        self.params_names = {"input", "name_"}
        self.name_ = self.__class__.__name__ # get_name(self)
        self.input_block = input_block
        self._input_block = _Input_Block_For_Class_Transparent_Block(self)
        if kargs:
            self.set_params(**kargs)

    def set_params(self, *args, **kargs):
        for v in args:
            self._set_param("input", v)
        for k,v in kargs.iteritems():
            self._set_param(k, v)

    def _set_param(self, k, v):
        if (k == "name_"):
            self.name_ = v
            # return True
        elif (k[:5] == "input"):
            self.input_block = v
            # return True
        elif isinstance(self.input_block, Base_Block):
            self.input_block.set_params(**{k:v})
            # elif isinstance(self.input_block, Base_Block) and self.input_block._set_param(k, v):
            # return True
        else:
            print("Warning : could not set attribute {} for block {}".format(k, self.name))
            # return False

    def get_params(self, *args, **kargs):
        if args:
            res = (self._get_param(k) for k in args)
            if kargs.has_key("as_dict") and kargs["as_dict"]:
                return {k:v for k,v in zip(args, res)}
            else:
                res = tuple(res)
                return (res[0] if (len(res) == 1) else res)
        else:
            return {k:self._get_param(k) for k in self.params_names}

    def _get_param(self, k):
        if (k[:5] == "input"):
            return self.input_block
        elif (k == "name_"):
            return self.name_
        else:
            return self.input_block._get_param(k)

    # def __getattribute__(self, attr):
    #     if (attr[:3] == "_id"):
    #         if isinstance(self.input_block, Base_Block):
    #             return self.input_block.__getattribute__(attr)
    #         else:
    #             return 2
    #     else:
    #         return super(Transparent_Block, self).__getattribute__(attr)
        
    def __getattr__(self, attr):
        if (attr[:3] == "_id"):
            if isinstance(self.input_block, Base_Block):
                return self.input_block.__getattribute__(attr)
            else:
                return 2
        if (attr[:4] == "set_"):
            return (lambda v : self.set_params(**{attr[4:]:v}))
        elif (attr[:4] == "get_"):
            return (lambda : self.get_params(attr[4:]))
        elif self.__dict__.has_key(attr) or (attr == "input_block"):
            return self.__dict__[attr]
        elif isinstance(self.input_block, Base_Block):
            return self.input_block.__getattribute__(attr)
            # return getattr(self.input_block, attr)
        elif (attr[:6] == "output"):
            return self.input_block
        else:
            raise AttributeError("Transparent Block could not find attribute {}".format(attr))

    def _input_block_(self):
        if isinstance(self.input_block, Base_Block):
            if (self._ct == 0):
                return self.input_block.train()
            elif (self._ct == 1):
                return self.input_block.test()
            else:
                return self.input_block()
        else:
            return self.input_block

    def train(self, *args, **kargs):
        self._ct = 0
        if args or kargs:
            self.set_params(*args, **kargs)
        return self._train()

    def _train(self):
        self.compute()
        
    def test(self, *args, **kargs):
        self._ct = 1
        if args or kargs:
            self.set_params(*args, **kargs)
        return self._test()

    def _test(self):
        self.compute()

    def __call__(self, *args, **kargs):
        self._ct = 2
        if args or kargs:
            self.set_params(*args, **kargs)
        return self._call()

    def _call(self):
        return self.compute()

    def compute(self):
        return self._input_block_()
