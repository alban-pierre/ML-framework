import time
from copy import copy, deepcopy

try:
    import signal
    from os import name as osname
    if (osname == "posix"):
        class TimeoutException(Exception): pass
except ImportError:
    TimeoutException = None

from base import *



class Measure_Time(Transparent_Block):
    """
    Measures time of computation of its input block
    The computation time is stored into "time" attribute
    Note that if input block is already computed, its time measured will be much smaller than expected
    Inputs :
        input_block : Base_Block = NoBlock : input block
        **kargs : **{} : Transparent_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that if you ask for "output" attribute, it redirects to input_block
    Examples :
        Measure_Time(in_block)
        Measure_Time(in_block, name="time_1")
    """
    def compute(self):
        self.time = -time.time()
        output = self._input_block_()
        self.time += time.time()
        return output
        


def Measure_Time_(in_block, *args, **kargs):
    raise NotImplementedError("Please use Measure_Time instead")



# def Measure_Time_(in_block, *args, **kargs):
#     if isinstance(in_block, Base_Block):
#         type_in_block = type(in_block)
#     else:
#         type_in_block = in_block
#     def time1(b):
#         if b.changed():
#             b.time = -time.time()
#             b.time_result = super(type_of_block, b).__call__()
#             b.time += time.time()
#         return b.time_result
#     type_of_block = type("Measure_Time", (type_in_block,), {"__call__":time1})
#     if isinstance(in_block, Base_Block):
#         inst = type_of_block.__new__(type_of_block)
#         for d in in_block.__dict__.keys():
#             setattr(inst, d, getattr(in_block, d))
#     else:
#         inst = type_of_block(*args, **kargs)
#     return inst



class Snapshot_Recursive(Transparent_Block):
    """
    Snapshot its input block and all of his inputs block as well
    Thus it takes a lot of memory space but you are sure to obtain the same result after (modulo seed)
    The snapshots are stored into "snapshots" attribute, which is a list of snapshots
    Inputs :
        input_block : Base_Block = NoBlock : input block
        **kargs : **{} : Transparent_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that if you ask for "output" attribute, it redirects to input_block
    Examples :
        Snapshot_Recursive(in_block)
        Snapshot_Recursive(in_block, name="snapshot_1")
    """
    # TODO: add max_nbr_snapshot attribute
    def __init__(self, *args, **kargs):
        super(Snapshot_Recursive, self).__init__(*args, **kargs)
        self.reset()

    def reset(self):
        self.snapshots = []

    def compute(self):
        output = self._input_block_()
        self.snapshots.append(deepcopy(self.input_block))
        return output



def Snapshot_Recursive_(in_block, *args, **kargs):
    raise NotImplementedError("Please use Snapshot_Recursive instead")



# def Snapshot_Recursive_(in_block, *args, **kargs):
#     if isinstance(in_block, Base_Block):
#         type_in_block = type(in_block)
#     else:
#         type_in_block = in_block
#     def snap1(b):
#         if b.changed():
#             b.snaprec_result = super(type_of_block, b).__call__()
#             snapshot = type_of_block.__new__(type_in_block)
#             for d in b.__dict__.keys():
#                 if (b != "snapshots"):
#                     setattr(snapshot, d, deepcopy(getattr(b, d)))
#             b.snapshots.append(snapshot)
#         return b.snaprec_result
#     def snap2(b):
#         b.snapshots = []
#     type_of_block = type("Snapshot_Recursive", (type_in_block,), {"__call__":snap1, "reset":snap2})
#     if isinstance(in_block, Base_Block):
#         inst = type_of_block.__new__(type_of_block)
#         for d in in_block.__dict__.keys():
#             setattr(inst, d, getattr(in_block, d))
#     else:
#         inst = type_of_block(*args, **kargs)
#     inst.reset()
#     return inst



class Snapshot_Block(Transparent_Block):
    """
    Snapshot its input block, but not recursively : its input_blocks are not copied
    The snapshots are stored into "snapshots" attribute, which is a list of snapshots
    Inputs :
        input_block : Base_Block = NoBlock : input block
        **kargs : **{} : Transparent_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that if you ask for "output" attribute, it redirects to input_block
    Examples :
        Snapshot_Block(in_block)
        Snapshot_Block(in_block, name="snapshot_1")
    """
    # TODO: add max_nbr_snapshot attribute
    def __init__(self, *args, **kargs):
        super(Snapshot_Block, self).__init__(*args, **kargs)
        self.reset()

    def reset(self):
        self.snapshots = []

    def compute(self):
        output = self._input_block_()
        snapshot = copy(self.input_block)
        if isinstance(snapshot, Base_Block):
            for k in snapshot.__dict__.keys():
                attr = getattr(snapshot, k)
                if not isinstance(attr, Base_Block):
                    setattr(snapshot, k, deepcopy(attr))
        else:
            snapshot = deepcopy(self.input_block)
        self.snapshots.append(snapshot)
        return output
        


def Snapshot_Block_(in_block, *args, **kargs):
    raise NotImplementedError("Please use Snapshot_Block instead")



# def Snapshot_Block_(in_block, *args, **kargs):
#     if isinstance(in_block, Base_Block):
#         type_in_block = type(in_block)
#     else:
#         type_in_block = in_block
#     def snap1(b):
#         if b.changed():
#             b.snapblock_result = super(type_of_block, b).__call__()
#             snapshot = type_of_block.__new__(type_in_block)
#             for d in b.__dict__.keys():
#                 if (b != "snapshots"):
#                     attr = getattr(b, d)
#                     if not isinstance(attr, Base_Block):
#                         setattr(snapshot, d, deepcopy(attr))
#             b.snapshots.append(snapshot)
#         return b.snapblock_result
#     def snap2(b):
#         b.snapshots = []
#     type_of_block = type("Snapshot_Block", (type_in_block,), {"__call__":snap1, "reset":snap2})
#     if isinstance(in_block, Base_Block):
#         inst = type_of_block.__new__(type_of_block)
#         for d in in_block.__dict__.keys():
#             setattr(inst, d, getattr(in_block, d))
#     else:
#         inst = type_of_block(*args, **kargs)
#     inst.reset()
#     return inst



class Snapshot_Output(Transparent_Block):
    """
    Snapshot the output of its input block
    The snapshots are stored into "snapshots", "snapshots_train" and "snapshots_test" attributes
        These attributes are lists of snapshots
    Inputs :
        input_block : Base_Block = NoBlock : input block
        **kargs : **{} : Transparent_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that if you ask for "output" attribute, it redirects to input_block
    Examples :
        Snapshot_Output(in_block)
        Snapshot_Output(in_block, name="snapshot_1")
    """
    # TODO: add max_nbr_snapshot attribute
    def __init__(self, *args, **kargs):
        super(Snapshot_Output, self).__init__(*args, **kargs)
        self.reset()

    def reset(self):
        self.snapshots_train = []
        self.snapshots_test = []
        self.snapshots = []

    def _train(self):
        output = self._input_block.train()
        self.snapshots_train.append(deepcopy(self.output_train))
        return output
        
    def _test(self):
        output = self._input_block.test()
        self.snapshots_test.append(deepcopy(self.output_test))
        return output

    def _call(self):
        output = self._input_block()
        self.snapshots.append(deepcopy(self.output))
        return output



def Snapshot_Output_(in_block, *args, **kargs):
    raise NotImplementedError("Please use Snapshot_Output instead")



# def Snapshot_Output_(in_block, *args, **kargs):
#     if isinstance(in_block, Base_Block):
#         type_in_block = type(in_block)
#     else:
#         type_in_block = in_block
#     def snap1(b):
#         if b.changed():
#             b.snapout_result = super(type_of_block, b).__call__()
#             b.snapshots.append(deepcopy(b.snapout_result))
#         return b.snapout_result
#     def snap2(b):
#         b.snapshots = []
#     type_of_block = type("Snapshot_Output", (type_in_block,), {"__call__":snap1, "reset":snap2})
#     if isinstance(in_block, Base_Block):
#         inst = type_of_block.__new__(type_of_block)
#         for d in in_block.__dict__.keys():
#             setattr(inst, d, getattr(in_block, d))
#     else:
#         inst = type_of_block(*args, **kargs)
#     inst.reset()
#     return inst


# TODO: add snapshot function, ie it runs a function on the input block and stores the result in snaps
class Snapshot_Attributes(Transparent_Block):
    """
    Snapshot some attributes of its input block
    The snapshots are stored into "snapshots" attributes, which is a list of dicts
    Inputs :
        input_block : Base_Block = NoBlock : input block
        attributes : str - [str] : the attributes names to snapshot
        **kargs : **{} : Transparent_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that if you ask for "output" attribute, it redirects to input_block
    Examples :
        Snapshot_Attributes(in_block)
        Snapshot_Attributes(in_block, name="snapshot_1")
        Snapshot_Attributes(in_block, attributes="seed")
        Snapshot_Attributes(in_block, attributes=["seed", "_ids"])
    """
    # TODO: add max_nbr_snapshot attribute
    def __init__(self, input_block=NoBlock, attributes=[], **kargs):
        super(Snapshot_Attributes, self).__init__(input_block)
        self.params_names = self.params_names.union({"attributes"})
        self.set_params(attributes=attributes, **kargs)
        self.reset()

    def _set_param(self, k, v):
        if (k == "attributes") or (k == "attrs"):
            if not isinstance(v, list):
                self.attributes = [v]
            else:
                self.attributes = v
        else:
            return super(Snapshot_Attributes, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "attributes") or (k == "attrs"):
            return self.attributes
        else:
            return super(Snapshot_Attributes, self)._get_param(k)

    def reset(self):
        self.snapshots = []

    def compute(self):
        output = self._input_block_()
        snapshot = {}
        if isinstance(self.input_block, Base_Block):
            for k in self.attributes:
                snapshot[k] = deepcopy(getattr(self.input_block, k))
        else:
            for k in self.attributes:
                if (k[:6] == "output"):
                    snapshot[k] = deepcopy(self.input_block)
        self.snapshots.append(snapshot)
        return output



def Snapshot_Attributes_(in_block, attributes, *args, **kargs):
    raise NotImplementedError("Please use Snapshot_Attributes instead")



# def Snapshot_Attributes_(in_block, attributes, *args, **kargs):
#     if not isinstance(attributes, list):
#         attributes = [attributes]
#     if isinstance(in_block, Base_Block):
#         type_in_block = type(in_block)
#     else:
#         type_in_block = in_block
#     def snap1(b):
#         if b.changed():
#             b.snapattr_result = super(type_of_block, b).__call__()
#             attrs = {}
#             for d in attributes:
#                 if (b != "snapshots"):
#                     attrs[d] = deepcopy(getattr(b, d))
#             b.snapshots.append(attrs)
#         return b.snapattr_result
#     def snap2(b):
#         b.snapshots = []
#     type_of_block = type("Snapshot_Attributes", (type_in_block,), {"__call__":snap1, "reset":snap2})
#     if isinstance(in_block, Base_Block):
#         inst = type_of_block.__new__(type_of_block)
#         for d in in_block.__dict__.keys():
#             setattr(inst, d, getattr(in_block, d))
#     else:
#         inst = type_of_block(*args, **kargs)
#     inst.reset()
#     return inst



class Run_Function_Before(Transparent_Block):
    """
    Run a function before calling its input block
    Inputs :
        input_block : Base_Block = NoBlock : input block
        func : ()->() = None : the function to call before input_block()
        **kargs : **{} : Transparent_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that if you ask for "output" attribute, it redirects to input_block
    Examples :
        Run_Function_Before(in_block)
        Run_Function_Before(in_block, name="func_1")
        Run_Function_Before(in_block, func=some_block.set_changed_here)
        Run_Function_Before(in_block, func=(lambda x=None : s.set_param("seed"=0)))
    """
    def __init__(self, input_block=NoBlock, func=None, **kargs):
        super(Run_Function_Before, self).__init__(input_block)
        self.params_names = self.params_names.union({"func"})
        self.set_params(func=func, **kargs)

    def _set_param(self, k, v):
        if (k == "function") or (k == "func"):
            self.func = v
        else:
            return super(Run_Function_Before, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "function") or (k == "func"):
            return self.func
        else:
            return super(Run_Function_Before, self)._get_param(k)

    def compute(self):
        self.func()
        return self._input_block_()



def Run_Function_Before_(in_block, func, *args, **kargs):
    raise NotImplementedError("Please use Run_Function_Before instead")



# def Run_Function_Before_(in_block, func, *args, **kargs):
#     if isinstance(in_block, Base_Block):
#         type_in_block = type(in_block)
#     else:
#         type_in_block = in_block
#     def time1(b):
#         if b.changed():
#             b.func()
#             b.time_result = super(type_of_block, b).__call__()
#         return b.time_result
#     type_of_block = type("Measure_Time", (type_in_block,), {"__call__":time1})
#     if isinstance(in_block, Base_Block):
#         inst = type_of_block.__new__(type_of_block)
#         for d in in_block.__dict__.keys():
#             setattr(inst, d, getattr(in_block, d))
#     else:
#         inst = type_of_block(*args, **kargs)
#     inst.func = func
#     return inst



class Run_Function_After(Transparent_Block):
    """
    Run a function after calling its input block
    Inputs :
        input_block : Base_Block = NoBlock : input block
        func : ()->() = None : the function to call after input_block()
        **kargs : **{} : Transparent_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that if you ask for "output" attribute, it redirects to input_block
    Examples :
        Run_Function_After(in_block)
        Run_Function_After(in_block, name="func_1")
        Run_Function_After(in_block, func=some_block.set_changed_here)
        Run_Function_After(in_block, func=(lambda x=None : s.set_param("seed"=0)))
    """
    def __init__(self, input_block=NoBlock, func=None, **kargs):
        super(Run_Function_After, self).__init__(input_block)
        self.params_names = self.params_names.union({"func"})
        self.set_params(func=func, **kargs)

    def _set_param(self, k, v):
        if (k == "function") or (k == "func"):
            self.func = v
        else:
            return super(Run_Function_After, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "function") or (k == "func"):
            return self.func
        else:
            return super(Run_Function_After, self)._get_param(k)

    def compute(self):
        output = self._input_block_()
        self.func()
        return output



def Run_Function_After_(in_block, func, *args, **kargs):
    raise NotImplementedError("Please use Run_Function_After instead")



# def Run_Function_After_(in_block, func, *args, **kargs):
#     if isinstance(in_block, Base_Block):
#         type_in_block = type(in_block)
#     else:
#         type_in_block = in_block
#     def time1(b):
#         if b.changed():
#             b.time_result = super(type_of_block, b).__call__()
#             b.func()
#         return b.time_result
#     type_of_block = type("Measure_Time", (type_in_block,), {"__call__":time1})
#     if isinstance(in_block, Base_Block):
#         inst = type_of_block.__new__(type_of_block)
#         for d in in_block.__dict__.keys():
#             setattr(inst, d, getattr(in_block, d))
#     else:
#         inst = type_of_block(*args, **kargs)
#     inst.func = func
#     return inst



class Verbose(Run_Function_After):
    """
    Same as Run_Function_After, it is used to print
    Inputs :
        input_block : Base_Block = NoBlock : input block
        func : ()->() = None : the function to call after input_block(), ie verbose function
        **kargs : **{} : Transparent_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that if you ask for "output" attribute, it redirects to input_block
    Examples :
        Verbose(in_block)
        Verbose(in_block, name="verb_1")
        def func1():
            print(t.time)
        Verbose(in_block, func=func1)
    """
    def __init__(self, input_block=NoBlock, func=None, name=None, **kargs):
        if name is None:
            name = "Verbose"
        super(Verbose, self).__init__(input_block, func=func, name_=name, **kargs)



class Force_Compute(Base_Input_Block):
    """
    Force the recomputation
    Every time this block output is required, it will act like the output is new
    So if you want to recompute Block b every time it is called, b input is this class
    I do not recommend to use this function, except at the end of a pipeline
    For example doing Force_Compute on a Split_Dataset is WRONG
    But you should do Force_Compute at the end on a Run_Function_Before that changes Split_Dataset seed
    Inputs :
        input_block : Base_Block = NoBlock : input block
        force : bool = True : set to False if you want to remove force without changing the pipeline
        **kargs : **{} : Base_Input_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that it stores result, not like other decorators blocks
    Examples :
        Force_Compute(in_block)
        Force_Compute(in_block, name="force_1")
        Force_Compute(in_block, force=False) # Useless, but you can set force to True later
    """
    def __init__(self, input_block=NoBlock, force=True, **kargs):
        super(Force_Compute, self).__init__(input_block)
        self.params_names = self.params_names.union({"force"})
        self.set_params(force=force, **kargs)

    def _set_param(self, k, v):
        if (k == "force_compute") or (k == "force"):
            self.force = v
            return True
        else:
            return super(Force_Compute, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "force_compute") or (k == "force"):
            return self.force
        else:
            return super(Force_Compute, self)._get_param(k)
    
    def changed_train(self):
        return self.force or super(Force_Compute, self).changed_train()
        
    def changed_test(self):
        return self.force or super(Force_Compute, self).changed_test()
        
    def changed_call(self):
        return self.force or super(Force_Compute, self).changed_call()
        
    def changed(self):
        return self.force or super(Force_Compute, self).changed()
        
    def compute(self):
        return self._input_block_()






class Timeout(Base_Input_Block):
    """
    Stop computation after n seconds, returning a default value if stopped
    It currently only work with linux
    Inputs :
        input_block : Base_Block = NoBlock : input block
        seconds : int = 10 : number of seconds to wait, after that we kill the input_block computation
        default_value : var = NoParam : the default value to return if we have killed input_block
        default_value : ()->var = NoParam : the default function to call if we have killed input_block
        **kargs : **{} : Base_Input_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that it stores result, not like other decorators blocks
            or self.default_value : if input_block computation is killed
            or self.default_func() : if input_block computation is killed and default_value is NoParam
    Examples :
        Force_Compute(in_block)
        Force_Compute(in_block, name="timeout_1")
        Force_Compute(in_block, seconds=5, default_value=0)
        Force_Compute(in_block, seconds=5, default_func=(lambda x=None : 0))
    """
    # TODO: Allow cascade of Timeout
    def __init__(self, input_block=NoBlock, seconds=10, default_value=NoParam, default_func=NoParam, **kargs):
        super(Timeout, self).__init__(input_block)
        self.params_names = self.params_names.union({"seconds", "default_value", "default_func"})
        self.set_params(seconds=seconds, default_value=default_value, default_func=default_func, **kargs)
        self._handler = None
        self._set_handler()
        if TimeoutException is None:
            print("Warning : Timeout could not be defined, either you are not on linux or the import failed")

    def set_params(self, *args, **kargs):
        super(Timeout, self).set_params(*args, **kargs)
        if (self.default_value is NoParam) and (self.default_func is NoParam):
            self.default_value = None

    def _set_param(self, k, v):
        if (k == "seconds") or (k == "sec") or (k == "time") or (k == "timeout"):
            self.seconds = v
            return True
        elif (k == "default") or (k == "default_value"):
            self.default_value = v
            return True
        elif (k == "default_func"):
            self.default_func = v
            return True
        else:
            return super(Timeout, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "seconds") or (k == "sec") or (k == "time") or (k == "timeout"):
            return self.seconds
        elif (k == "default") or (k == "default_value"):
            return self.default_value
        elif (k == "default_func"):
            return self.default_func
        else:
            return super(Timeout, self)._get_param(k)

    def _set_handler(self):
        if (self._handler is None) and (TimeoutException is not None):
            def _handler(signum, frame):
                raise TimeoutException
            self._handler = _handler
            signal.signal(signal.SIGALRM, self._handler)

    def compute(self):
        if (TimeoutException is None):
            output = self._input_block_()
        else:
            signal.alarm(self.seconds)
            try:
                output = self._input_block_()
            except TimeoutException:
                if (self.default_value is NoParam):
                    output = self.default_func()
                elif isinstance(self.default_value, Exception):
                    raise self.default_value
                else:
                    output = self.default_value
            except:
                signal.alarm(0)
                raise
            signal.alarm(0)
        return output



class Ignore_Exception(Base_Input_Block):
    """
    Catch exceptions, returning a default value if exception is catched
    Inputs :
        input_block : Base_Block = NoBlock : input block
        exceptions : exception - [exception] = [] : exceptions to catch, returning default value/func
        default_value : var = NoParam : the default value to return if we have killed input_block
        default_value : ()->var = NoParam : the default function to call if we have killed input_block
        **kargs : **{} : Base_Input_Block parameters can be set here
    Outputs : 
        self.input_block() : Note that it stores result, not like other decorators blocks
            or self.default_value : if input_block computation is killed
            or self.default_func() : if input_block computation is killed and default_value is NoParam
    Examples :
        Ignore_Exception(in_block)
        Ignore_Exception(in_block, name="except_1")
        Ignore_Exception(in_block, exceptions=ValueError, default_value=0)
        Ignore_Exception(in_block, exceptions=[ValueError, OSError], default_value=0)
        Ignore_Exception(in_block, exceptions=[ValueError, OSError], default_func=(lambda x=None : 0))
    """
    def __init__(self, input_block=NoBlock, exceptions=[], default_value=NoParam, default_func=NoParam, **kargs):
        super(Ignore_Exception, self).__init__(input_block)
        self.exceptions = []
        self.set_params(exceptions=exceptions, default_value=default_value, default_func=default_func, **kargs)

    def set_params(self, *args, **kargs):
        super(Ignore_Exception, self).set_params(*args, **kargs)
        if (self.default_value is NoParam) and (self.default_func is NoParam):
            self.default_value = None

    def _set_param(self, k, v):
        if (k == "default") or (k == "default_value"):
            self.default_value = v
            return True
        elif (k == "default_func"):
            self.default_func = v
            return True
        elif (k == "exceptions") or (k == "exception"):
            if isinstance(v, list):
                self.exceptions = v
            else:
                self.exceptions.append(v)
            return True
        else:
            return super(Ignore_Exception, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "default") or (k == "default_value"):
            return self.default_value
        elif (k == "default_func"):
            return self.default_func
        elif (k == "exceptions") or (k == "exception"):
            return self.exceptions
        else:
            return super(Ignore_Exception, self)._get_param(k)

    def _is_exception_instance(self, e):
        for exc in self.exceptions:
            if isinstance(e, exc):
                return True
        return False

    def compute(self):
        try:
            output = self._input_block_()
        except Exception as e:
            if self._is_exception_instance(e):
                if (self.default_value is NoParam):
                    output = self.default_func()
                elif isinstance(self.default_value, Exception):
                    raise self.default_value
                else:
                    output = self.default_value
            else:
                raise e
        return output
